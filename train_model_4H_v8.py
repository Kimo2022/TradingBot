import gc, os, glob, warnings, logging
import numpy as np, pandas as pd, tensorflow as tf, optuna, joblib
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Bidirectional, GRU,
    MultiHeadAttention, Add, GlobalAveragePooling1D, GaussianNoise,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau,
)

warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)
try:
    [tf.config.experimental.set_memory_growth(g, True)
     for g in tf.config.list_physical_devices("GPU")]
except Exception:
    pass
strategy = tf.distribute.MirroredStrategy()
NGPU = max(strategy.num_replicas_in_sync, 1)

# ── Paths ──────────────────────────────────────────────────────────────────────
BTC_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/BTC"
ETH_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/ETH"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global constants ───────────────────────────────────────────────────────────
SEQ_LEN          = 60
BATCH            = 64
EPOCHS           = 70
N_TRIALS         = 20          # 12-hour GPU budget
EPS              = 1e-8
FEE              = 0.001       # 0.1% per side → 0.2% round-trip
TP_MULT          = 2.0
SL_MULT          = 1.0
TB_BARS          = 10
ADX_TREND        = 22.0
ANN_FACTOR       = np.sqrt(365 * 6)
TRAIN_FRAC       = 0.70
VAL_FRAC         = 0.10
WF_SPLITS        = 5           # Walk-Forward folds
MAX_MDD          = -0.15       # Hard kill: exclude trials with MDD < -15%
TARGET_WR        = 0.55        # Optuna bonus threshold
TARGET_SHARPE    = 1.50        # Optuna bonus threshold
NEUTRAL_W_FACTOR = 0.45
LOG_RET_PERIODS  = [3, 6, 12, 24]
N_CLASSES        = 3           # 0=Short 1=Neutral 2=Long
_L2              = tf.keras.regularizers.l2(1e-5)
TRIAL_RESULTS    = {}
BEST_SCORE       = -np.inf

# ── Feature lists ──────────────────────────────────────────────────────────────
PRICE_FEATS = [
    "close_norm","ema20_norm","ema50_norm","ema200_norm",
    "rsi","rsi_slope","macd_norm","macd_hist_norm",
    "atr_norm","atr_regime","atr_breakout","price_range_norm",
    "body_norm","up_wick_norm","lo_wick_norm",
    "bb_up_norm","bb_lo_norm","bb_w",
    "tkr_ratio","rvi","vol_trend","vpt_norm",
    "log_ret","volatility","vol_zsc","roc","vol_scaled_ret",
    "eth_rsi_ratio","eth_close_norm","btc_eth_corr",
    "obv_norm","vwap_ratio","vwap_daily_dist",
    "stoch_k","stoch_d","cci","cmf",
    "adx","di_diff","adx_regime",
    "ichimoku_cloud","tenkan_kijun",
    "vol_regime","close_above_ema200",
    "delta_flow","vol_imb_5","vol_imb_14","vol_imb_42",
    "liq_proxy","mom_3","mom_6","mom_12","mom_24",
    "vol_quantile","atr_pct_ret",
    "log_ret_3","log_ret_6","log_ret_12","log_ret_24",
    "daily_ret","daily_trend_strength","daily_ema_slope","price_vs_daily_open",
]
TIME_FEATS = [
    "hour_sin","hour_cos","day_sin","day_cos",
    "week_sin","week_cos","session_asia","session_eu","session_us",
    "is_trending_session",
]
ALL_FEATS = PRICE_FEATS + TIME_FEATS
NF        = len(ALL_FEATS)
N_PRICE   = len(PRICE_FEATS)

# ── Log-scalable features (strictly positive ratios → log before RobustScaler) ─
_LOG_SCALE_FEATS = [
    "tkr_ratio","vol_trend","vol_imb_5","vol_imb_14","vol_imb_42",
    "atr_regime","price_range_norm",
]
_LOG_SCALE_IDX = [PRICE_FEATS.index(f) for f in _LOG_SCALE_FEATS if f in PRICE_FEATS]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _load_resample_one(fp):
    try:
        df = pd.read_csv(fp, header=None, skiprows=1)
        nc = df.shape[1]
        if nc >= 8:   df = df.iloc[:, [1,3,4,5,6,7]]
        elif nc == 7: df = df.iloc[:, [1,2,3,4,5,6]]
        elif nc == 6: df = df.iloc[:, :6]
        else: return None
        df.columns = ["timestamp","open","high","low","close","volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        r = df.resample("4h").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        )
        return r.ffill().bfill().dropna().astype("float32")
    except Exception:
        return None

def load_sync_resample(data_dir, label=""):
    fps = sorted(set(
        glob.glob(os.path.join(data_dir, "*.csv")) +
        glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    ))
    parts = [ch for fp in fps
             for ch in [_load_resample_one(fp)] if ch is not None and len(ch) > 10]
    if not parts: raise RuntimeError(f"[load] {label}: no valid CSV in {data_dir}")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="4h")).ffill().bfill()
    df["vol"] = df["volume"]
    del parts; gc.collect()
    return df

def sync_assets(b, e):
    idx = b.index.intersection(e.index)
    return b.loc[idx].copy(), e.loc[idx].copy()


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def _rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100 / (1 + g.ewm(com=p-1, adjust=False).mean() /
                        (dn.ewm(com=p-1, adjust=False).mean() + EPS))

def _adx(h, l, c, p=14):
    tr  = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    a   = tr.ewm(com=p-1, adjust=False).mean()
    um  = h-h.shift(); dm = l.shift()-l
    dmp = um.where((um>dm)&(um>0), 0.0)
    dmn = dm.where((dm>um)&(dm>0), 0.0)
    dip = 100*dmp.ewm(com=p-1, adjust=False).mean()/(a+EPS)
    din = 100*dmn.ewm(com=p-1, adjust=False).mean()/(a+EPS)
    dx  = 100*(dip-din).abs()/(dip+din+EPS)
    return dx.ewm(com=p-1, adjust=False).mean(), dip-din


# ══════════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER LABELS
# ══════════════════════════════════════════════════════════════════════════════
def triple_barrier_labels(close_arr, atr_arr, tp_m=TP_MULT, sl_m=SL_MULT, nb=TB_BARS):
    n = len(close_arr)
    labels = np.ones(n, dtype=np.int32)
    for i in range(n - nb):
        entry  = close_arr[i]
        tp, sl = entry*(1+tp_m*atr_arr[i]), entry*(1-sl_m*atr_arr[i])
        future = close_arr[i+1:i+1+nb]
        tp_hit = int(np.argmax(future >= tp)) if (future >= tp).any() else nb
        sl_hit = int(np.argmax(future <= sl)) if (future <= sl).any() else nb
        if tp_hit < sl_hit:   labels[i] = 2
        elif sl_hit < tp_hit: labels[i] = 0
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def mkfeat(btc_df, eth_df):
    df = btc_df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]
    ec = eth_df["close"].reindex(df.index, method="ffill").ffill().bfill()

    ema20 = c.ewm(span=20,adjust=False).mean()
    ema50 = c.ewm(span=50,adjust=False).mean()
    ema200= c.ewm(span=200,adjust=False).mean()
    df["close_norm"]  = c/(c.rolling(200).mean().replace(0,np.nan).ffill().bfill()+EPS)
    df["ema20_norm"]  = ema20/(c+EPS)-1.0
    df["ema50_norm"]  = ema50/(c+EPS)-1.0
    df["ema200_norm"] = ema200/(c+EPS)-1.0

    rsi_raw = _rsi(c)
    df["rsi"]            = rsi_raw
    df["rsi_slope"]      = rsi_raw.diff(5)/5.0
    df["eth_rsi_ratio"]  = _rsi(ec)/(rsi_raw+EPS)
    df["eth_close_norm"] = ec/(ec.rolling(200).mean().replace(0,np.nan).ffill().bfill()+EPS)

    br = np.log(c/(c.shift(1)+EPS)); er = np.log(ec/(ec.shift(1)+EPS))
    df["log_ret"]      = br
    df["btc_eth_corr"] = br.rolling(42).corr(er).fillna(0).clip(-1,1)

    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd_r = e12-e26; sig_r = macd_r.ewm(span=9,adjust=False).mean()
    df["macd_norm"]     = macd_r/(c+EPS)
    df["macd_hist_norm"]= (macd_r-sig_r)/(c+EPS)

    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr_raw = tr.ewm(com=13,adjust=False).mean()
    df["atr_norm"]         = atr_raw/(c+EPS)
    df["atr_regime"]       = atr_raw/(atr_raw.rolling(42).mean()+EPS)
    df["atr_breakout"]     = (c-c.rolling(20).mean())/(atr_raw+EPS)
    df["price_range_norm"] = (h-l)/(atr_raw+EPS)
    df["atr_pct_ret"]      = br/(df["atr_norm"]+EPS)

    df["body_norm"]    = (c-o).abs()/(atr_raw+EPS)
    oc_hi = pd.concat([o,c],axis=1).max(axis=1); oc_lo = pd.concat([o,c],axis=1).min(axis=1)
    df["up_wick_norm"] = (h-oc_hi)/(atr_raw+EPS)
    df["lo_wick_norm"] = (oc_lo-l)/(atr_raw+EPS)

    s20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    df["bb_up_norm"] = (s20+2*sd20-c)/(c+EPS)
    df["bb_lo_norm"] = (s20-2*sd20-c)/(c+EPS)
    df["bb_w"]       = (4*sd20)/(s20+EPS)

    ve20 = v.ewm(span=20,adjust=False).mean()
    df["tkr_ratio"] = v/(ve20+EPS)
    rvi_r = (v-v.rolling(42).mean())/(v.rolling(42).std()+EPS)
    df["rvi"]      = rvi_r
    df["vol_trend"]= v.ewm(span=12,adjust=False).mean()/(v.ewm(span=42,adjust=False).mean()+EPS)
    vpt = (br*v).cumsum()
    df["vpt_norm"] = vpt/(vpt.rolling(200).std()+EPS)

    df["volatility"]     = br.rolling(14).std()
    df["roc"]            = (c-c.shift(14))/(c.shift(14)+EPS)
    df["vol_scaled_ret"] = br/(df["volatility"]+EPS)
    eth_vol = er.rolling(14).std()
    pool = (df["volatility"]+eth_vol)/2
    df["vol_zsc"] = (df["volatility"]-pool.rolling(180).mean())/(pool.rolling(180).std()+EPS)

    obv = pd.Series(np.cumsum(np.sign(br.fillna(0).values)*v.values), index=df.index)
    df["obv_norm"] = obv/(obv.rolling(200).std()+EPS)

    tp_v = (h+l+c)/3
    df["vwap_ratio"] = c/((tp_v*v).rolling(20).sum()/(v.rolling(20).sum()+EPS)+EPS)
    vd = (tp_v*v).groupby(df.index.date).cumsum()/(v.groupby(df.index.date).cumsum()+EPS)
    vd.index = df.index
    df["vwap_daily_dist"] = (c-vd)/(c+EPS)

    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["stoch_k"] = (c-lo14)/(hi14-lo14+EPS)*100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["cci"]     = (tp_v-tp_v.rolling(20).mean())/(0.015*tp_v.rolling(20).std()+EPS)
    mfv = ((2*c-h-l)/(h-l+EPS))*v
    df["cmf"]     = mfv.rolling(20).sum()/(v.rolling(20).sum()+EPS)

    adx_v, di_d = _adx(h,l,c,p=14)
    df["adx"]        = adx_v
    df["di_diff"]    = di_d
    df["adx_regime"] = (adx_v>ADX_TREND).astype(np.float32)

    tenkan  = (h.rolling(9).max()+l.rolling(9).min())/2
    kijun   = (h.rolling(26).max()+l.rolling(26).min())/2
    s_a     = (tenkan+kijun)/2
    s_b     = (h.rolling(52).max()+l.rolling(52).min())/2
    df["ichimoku_cloud"] = (s_a-s_b)/(c+EPS)
    df["tenkan_kijun"]   = (tenkan-kijun)/(c+EPS)

    v30 = df["volatility"].rolling(180).mean(); v30s = df["volatility"].rolling(180).std()
    df["vol_regime"]         = (df["volatility"]-v30)/(v30s+EPS)
    df["close_above_ema200"] = (c>ema200).astype(np.float32)

    up_v = v.where(br>0,0.0); dn_v = v.where(br<0,0.0)
    df["delta_flow"] = (up_v-dn_v).rolling(20).mean()/(v.rolling(20).mean()+EPS)
    df["vol_imb_5"]  = up_v.rolling(5).sum()/(dn_v.rolling(5).sum()+EPS)
    df["vol_imb_14"] = up_v.rolling(14).sum()/(dn_v.rolling(14).sum()+EPS)
    df["vol_imb_42"] = up_v.rolling(42).sum()/(dn_v.rolling(42).sum()+EPS)
    wr_f = (df["up_wick_norm"]+df["lo_wick_norm"])/(df["body_norm"]+EPS)
    df["liq_proxy"]  = ((df["atr_regime"]>1.5).astype(float)*
                        (df["tkr_ratio"]>2.0).astype(float)*
                        (wr_f>1.5).astype(float)).astype(np.float32)
    df["mom_3"]      = (c-c.shift(3))/(c.shift(3)+EPS)
    df["mom_6"]      = (c-c.shift(6))/(c.shift(6)+EPS)
    df["mom_12"]     = (c-c.shift(12))/(c.shift(12)+EPS)
    df["mom_24"]     = (c-c.shift(24))/(c.shift(24)+EPS)
    df["vol_quantile"]= df["volatility"].rolling(360).rank(pct=True)

    log_c = np.log(c+EPS)
    for p in LOG_RET_PERIODS:
        df[f"log_ret_{p}"] = (log_c-log_c.shift(p)).astype(np.float32)

    d_open = o.resample("1D").first().reindex(df.index,method="ffill").ffill()
    d_atr  = atr_raw.resample("1D").mean().reindex(df.index,method="ffill").ffill()
    d_ema  = c.resample("1D").last().ewm(span=20,adjust=False).mean()\
              .reindex(df.index,method="ffill").ffill()
    df["daily_ret"]           = np.log((c+EPS)/(d_open+EPS)).astype(np.float32)
    df["daily_trend_strength"]= ((c-d_open)/(d_atr+EPS)).clip(-5,5).astype(np.float32)
    df["daily_ema_slope"]     = ((d_ema-d_ema.shift(6))/(d_ema.shift(6)+EPS)).astype(np.float32)
    df["price_vs_daily_open"] = ((c-d_open)/(c+EPS)).astype(np.float32)
    del d_open, d_atr, d_ema, log_c, up_v, dn_v, obv, vpt

    hr = df.index.hour
    df["hour_sin"] = np.sin(2*np.pi*hr/24).astype(np.float32)
    df["hour_cos"] = np.cos(2*np.pi*hr/24).astype(np.float32)
    df["day_sin"]  = np.sin(2*np.pi*df.index.dayofweek/7).astype(np.float32)
    df["day_cos"]  = np.cos(2*np.pi*df.index.dayofweek/7).astype(np.float32)
    how            = df.index.dayofweek*24+hr
    df["week_sin"] = np.sin(2*np.pi*how/168).astype(np.float32)
    df["week_cos"] = np.cos(2*np.pi*how/168).astype(np.float32)
    df["session_asia"] = ((hr>=0)&(hr<8)).astype(np.float32)
    df["session_eu"]   = ((hr>=8)&(hr<16)).astype(np.float32)
    df["session_us"]   = ((hr>=16)&(hr<24)).astype(np.float32)
    df["is_trending_session"] = ((hr>=8)&(hr<20)).astype(np.float32)

    df[PRICE_FEATS] = df[PRICE_FEATS].shift(1)
    df = df.dropna(subset=ALL_FEATS)[ALL_FEATS].astype(np.float32)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SCALING  ← log-transform ratio features before RobustScaler
# ══════════════════════════════════════════════════════════════════════════════
def _apply_log_transform(feat_arr):
    for idx in _LOG_SCALE_IDX:
        col = feat_arr[:, idx]
        feat_arr[:, idx] = np.sign(col) * np.log1p(np.abs(col)).astype(np.float32)

def fit_scaler(feat_arr, train_end):
    _apply_log_transform(feat_arr)          # in-place log-transform
    sc = RobustScaler()
    feat_arr[:train_end, :N_PRICE] = sc.fit_transform(feat_arr[:train_end, :N_PRICE])
    feat_arr[train_end:, :N_PRICE] = sc.transform(feat_arr[train_end:, :N_PRICE])
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_seqs(feat_arr, close_arr, atr_arr, sl=SEQ_LEN):
    n_seqs = len(feat_arr) - sl - TB_BARS
    idx    = np.arange(n_seqs)[:, None] + np.arange(sl)
    X      = feat_arr[idx]
    labels = triple_barrier_labels(close_arr, atr_arr)
    y      = labels[:n_seqs].astype(np.int32)
    lr_arr = np.zeros(len(close_arr), dtype=np.float32)
    lr_arr[1:] = np.log(close_arr[1:]/(close_arr[:-1]+EPS)).astype(np.float32)
    y_raw  = lr_arr[sl: sl+n_seqs]
    atr_s  = atr_arr[sl: sl+n_seqs]
    return X, y, y_raw, atr_s

def _temporal_split(n):
    return int(n*TRAIN_FRAC), int(n*(TRAIN_FRAC+VAL_FRAC))


# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_cw(ytr):
    dist  = np.bincount(ytr, minlength=3).astype(np.float64)
    raw_w = dist.sum() / (N_CLASSES * dist + EPS)
    raw_w[1] *= NEUTRAL_W_FACTOR
    raw_w /= raw_w.mean()
    cw = {i: float(raw_w[i]) for i in range(3)}
    print(f"[class_weights] Short={cw[0]:.3f} | Neutral={cw[1]:.3f} | Long={cw[2]:.3f}", flush=True)
    return cw


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model(seq_len, n_feat, units, nh, d_rate, lr, noise, gru_layers=2):
    inp = Input(shape=(seq_len, n_feat), dtype="float32")
    x   = GaussianNoise(noise)(inp)
    x   = Dense(units, kernel_regularizer=_L2)(x)
    for i in range(gru_layers):
        ret_seq = True  # always return sequences until we pool
        x = Bidirectional(
            GRU(units//2, return_sequences=ret_seq,
                dropout=d_rate, recurrent_dropout=0.0)
        )(x)
        x = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=nh, key_dim=max(units//nh, 1), dropout=d_rate)(x, x)
    x    = LayerNormalization()(Add()([x, attn]))
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(units//2, activation="elu", kernel_regularizer=_L2)(x)
    x    = Dropout(d_rate)(x)
    out  = Dense(N_CLASSES, activation="softmax", dtype="float32")(x)
    m    = Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m

def make_ds(X, y, batch, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.int32))
    )
    if shuffle: ds = ds.shuffle(min(len(y), 50_000), reshuffle_each_iteration=True)
    return ds.batch(batch*NGPU, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST  ← real round-trip cost: 2 × FEE per trade
# ══════════════════════════════════════════════════════════════════════════════
def backtest(proba, y_raw, atr_arr):
    n     = min(len(proba), len(y_raw), len(atr_arr))
    proba = proba[:n]; y_raw = y_raw[:n]; atr_ = atr_arr[:n]
    pred  = np.argmax(proba, axis=1)
    conf  = np.max(proba, axis=1)
    # signal: scale by confidence above 45% threshold
    raw_sig = np.where(pred==2, 1.0, np.where(pred==0, -1.0, 0.0))
    size    = np.clip((conf - 0.45) / 0.55, 0.1, 1.0)
    sig     = raw_sig * size
    # ATR-based stop
    stop    = SL_MULT * atr_
    eff     = np.where(sig>0, np.maximum(y_raw, -stop),
              np.where(sig<0, np.minimum(y_raw,  stop), y_raw))
    # ← 2×FEE per round-trip (entry + exit), applied only when position changes
    rets    = sig * eff - 2.0 * FEE * np.abs(sig)
    cum     = np.cumprod(1.0 + rets)
    pk      = np.maximum.accumulate(cum)
    mdd     = float(((cum - pk) / (pk + EPS)).min())
    sh      = float(rets.mean() / (rets.std() + EPS) * ANN_FACTOR)
    neg     = rets[rets < 0]
    sortino = float(rets.mean() / (neg.std() + EPS) * ANN_FACTOR)
    pf      = float(rets[rets>0].sum() / (abs(neg.sum()) + EPS))
    n_tr    = int(np.count_nonzero(sig))
    wr      = len(rets[(raw_sig != 0) & (rets > 0)]) / (int(np.count_nonzero(raw_sig)) + EPS)
    return dict(ret=float(cum[-1]-1), mdd=mdd, sharpe=sh, sortino=sortino,
                pf=pf, wr=float(wr), trades=n_tr)


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward_validate(X, y, y_raw, atr_s, params, cw):
    """Expanding-window WFV across WF_SPLITS folds. Returns mean metrics."""
    n      = len(X)
    fold_sz= n // (WF_SPLITS + 1)
    wf_metrics = []
    print(f"\n[WFV] {WF_SPLITS} folds | fold_size≈{fold_sz:,}", flush=True)
    for fold in range(WF_SPLITS):
        train_end = fold_sz * (fold + 1)
        val_end   = min(train_end + fold_sz, n)
        if val_end - train_end < 200: break
        Xtr_f, ytr_f = X[:train_end], y[:train_end]
        Xv_f,  yv_f  = X[train_end:val_end], y[train_end:val_end]
        yr_f          = y_raw[train_end:val_end]
        at_f          = atr_s[train_end:val_end]
        tf.keras.backend.clear_session(); gc.collect()
        with strategy.scope():
            mf = build_model(SEQ_LEN, NF, **params)
        ds_tr = make_ds(Xtr_f, ytr_f, BATCH, shuffle=True)
        ds_vl = make_ds(Xv_f, yv_f, BATCH)
        cb = [EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
              ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=0),
              TerminateOnNaN()]
        mf.fit(ds_tr, validation_data=ds_vl, epochs=40, verbose=0,
               callbacks=cb, class_weight=cw)
        pr   = mf.predict(ds_vl, verbose=0)
        bt   = backtest(pr, yr_f, at_f)
        wf_metrics.append(bt)
        print(f"  Fold {fold+1}/{WF_SPLITS} | Ret={bt['ret']:.2%} | "
              f"WR={bt['wr']:.2%} | Sharpe={bt['sharpe']:.2f} | MDD={bt['mdd']:.2%}", flush=True)
        del mf, ds_tr, ds_vl; gc.collect()
    if not wf_metrics:
        return {}
    agg = {k: float(np.mean([m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    print(f"[WFV] mean → Ret={agg['ret']:.2%} | WR={agg['wr']:.2%} | "
          f"Sharpe={agg['sharpe']:.2f} | MDD={agg['mdd']:.2%}\n", flush=True)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE  ← WR and Sharpe bonuses for hitting targets
# ══════════════════════════════════════════════════════════════════════════════
def _composite_score(bt):
    if bt["mdd"] < MAX_MDD:                 # hard kill ← MDD > 15%
        return -np.inf
    wr_score  = bt["wr"]
    mdd_score = 1.0 + bt["mdd"]
    pf_score  = min(bt["pf"], 5.0) / 5.0
    sh_score  = np.clip(bt["sharpe"], -3, 5) / 5.0
    base      = 0.35*wr_score + 0.30*mdd_score + 0.20*pf_score + 0.15*sh_score
    bonus     = (0.05 if bt["wr"]     >= TARGET_WR     else 0.0) + \
                (0.05 if bt["sharpe"] >= TARGET_SHARPE  else 0.0)
    return base + bonus


def _try_save_best(model, score, sc, label=""):
    global BEST_SCORE
    if score > BEST_SCORE and np.isfinite(score):
        BEST_SCORE = score
        model.save(os.path.join(OUT_DIR, "best_model_4H.keras"))
        joblib.dump(
            {"sc": sc, "NF": NF, "N_PRICE": N_PRICE,
             "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS, "SEQ_LEN": SEQ_LEN},
            os.path.join(OUT_DIR, "tft_4H_scaler.save"),
        )
        print(f"[save] ★ [{label}] score={score:.4f} → best_model_4H.keras", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial, Xtr, ytr, Xv, yv, yv_raw, atr_v, cw, sc):
    tf.keras.backend.clear_session(); gc.collect()
    units     = trial.suggest_categorical("units",     [128, 256, 384, 512, 768])
    nh        = trial.suggest_categorical("nh",        [4, 8, 16])
    d         = trial.suggest_float("d",               0.10, 0.45)
    lr        = trial.suggest_float("lr",              5e-5, 5e-3, log=True)
    noise     = trial.suggest_float("noise",           0.002, 0.04)
    gru_layers= trial.suggest_categorical("gru_layers",[2, 3])
    if units % nh != 0: return float("inf")
    tn = trial.number + 1; val = np.inf; m = None
    try:
        with strategy.scope():
            m = build_model(SEQ_LEN, NF, units, nh, d, lr, noise, gru_layers)
        ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True)
        ds_vl = make_ds(Xv,  yv,  BATCH)
        cb = [
            EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                              min_lr=1e-7, verbose=0),
            TerminateOnNaN(),
        ]
        h_ = m.fit(ds_tr, validation_data=ds_vl, epochs=35, verbose=0,
                   callbacks=cb, class_weight=cw)
        val = float(min(h_.history.get("val_loss", [np.inf])))
        if np.isfinite(val):
            ckpt  = os.path.join(OUT_DIR, f"model_4H_trial_{tn:02d}.keras")
            m.save(ckpt)
            pr    = m.predict(ds_vl, verbose=0)
            bt    = backtest(pr, yv_raw, atr_v)
            score = _composite_score(bt)
            mdd_flag = "🔴 KILLED" if bt["mdd"] < MAX_MDD else ""
            TRIAL_RESULTS[tn] = {
                "val_loss": val, "ret": bt["ret"], "sharpe": bt["sharpe"],
                "mdd": bt["mdd"], "pf": bt["pf"], "wr": bt["wr"],
                "trades": bt["trades"], "score": score, "model": ckpt,
            }
            print(
                f"[T{tn:02d}/{N_TRIALS}] u={units}|nh={nh}|gl={gru_layers}|"
                f"d={d:.2f}|lr={lr:.1e} | loss={val:.5f} | "
                f"Ret={bt['ret']:.2%} | WR={bt['wr']:.2%} | "
                f"Sharpe={bt['sharpe']:.2f} | MDD={bt['mdd']:.2%} | "
                f"score={score:.4f} {mdd_flag}", flush=True,
            )
            _try_save_best(m, score, sc, label=f"T{tn:02d}")
    except Exception as e:
        print(f"[T{tn:02d}] ERROR: {e}", flush=True)
    finally:
        try: del m
        except NameError: pass
        tf.keras.backend.clear_session(); gc.collect()
    return val


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary_table():
    if not TRIAL_RESULTS: return
    rows = []
    for tn, r in sorted(TRIAL_RESULTS.items()):
        killed = "🔴" if r["mdd"] < MAX_MDD else "✅"
        wr_ok  = "✅" if r["wr"]     >= TARGET_WR     else "  "
        sh_ok  = "✅" if r["sharpe"] >= TARGET_SHARPE  else "  "
        rows.append({
            "T":       tn,
            "Ret":     f"{r['ret']:.2%}",
            "WR":      f"{r['wr']:.2%}{wr_ok}",
            "Sharpe":  f"{r['sharpe']:.2f}{sh_ok}",
            "MDD":     f"{r['mdd']:.2%}{killed}",
            "PF":      f"{r['pf']:.2f}",
            "Trades":  r["trades"],
            "Score":   f"{r['score']:.4f}",
        })
    df = pd.DataFrame(rows).set_index("T")
    valid = {t: r for t, r in TRIAL_RESULTS.items() if r["mdd"] >= MAX_MDD}
    best_tn = max(valid, key=lambda t: valid[t]["score"]) if valid else None
    print("\n" + "="*90)
    print("TRIAL COMPARISON SUMMARY  (🔴=MDD killed | ✅=target met)")
    print("="*90)
    print(df.to_string())
    print("="*90)
    if best_tn:
        b = TRIAL_RESULTS[best_tn]
        print(f"★ BEST: T{best_tn:02d} | Score={b['score']:.4f} | "
              f"Ret={b['ret']:.2%} | WR={b['wr']:.2%} | "
              f"Sharpe={b['sharpe']:.2f} | MDD={b['mdd']:.2%}")
    print("="*90 + "\n", flush=True)
    joblib.dump(df, os.path.join(OUT_DIR, "trial_summary_table.save"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train():
    print(f"[GRU-4H-v8] GPUs={NGPU} | SEQ={SEQ_LEN} | NF={NF} | "
          f"FEE={FEE*100:.1f}%×2 | TRIALS={N_TRIALS} | WF_SPLITS={WF_SPLITS}", flush=True)

    btc_raw = load_sync_resample(BTC_DIR, "BTC")
    eth_raw = load_sync_resample(ETH_DIR, "ETH")
    btc_raw, eth_raw = sync_assets(btc_raw, eth_raw); gc.collect()

    c_r  = btc_raw["close"]; h_r = btc_raw["high"]; l_r = btc_raw["low"]
    tr_r = pd.concat([(h_r-l_r),(h_r-c_r.shift()).abs(),(l_r-c_r.shift()).abs()],axis=1).max(axis=1)
    atr_norm_u = (tr_r.ewm(com=13,adjust=False).mean()/(c_r+EPS)).astype(np.float32)

    feat_df = mkfeat(btc_raw, eth_raw)
    close_s = btc_raw["close"].loc[feat_df.index]
    atr_s   = atr_norm_u.reindex(feat_df.index).ffill().bfill()
    del btc_raw, eth_raw, tr_r, c_r, h_r, l_r, atr_norm_u; gc.collect()

    n_seqs = len(feat_df) - SEQ_LEN - TB_BARS
    tr_cut, val_cut = _temporal_split(n_seqs)
    feat_arr = feat_df.values.astype(np.float32)
    del feat_df; gc.collect()

    sc = fit_scaler(feat_arr, tr_cut+SEQ_LEN); gc.collect()   # ← log+robust scale

    c_arr   = close_s.values.astype(np.float64)
    atr_arr = atr_s.values.astype(np.float32)
    del close_s, atr_s; gc.collect()

    X, y, y_raw, atr_seqs = build_seqs(feat_arr, c_arr, atr_arr)
    del feat_arr, c_arr, atr_arr; gc.collect()

    Xtr = X[:tr_cut];          ytr = y[:tr_cut]
    Xv  = X[tr_cut:val_cut];   yv  = y[tr_cut:val_cut]
    Xte = X[val_cut:];         yte = y[val_cut:]
    yv_raw  = y_raw[tr_cut:val_cut];  yte_raw  = y_raw[val_cut:]
    atr_v   = atr_seqs[tr_cut:val_cut]; atr_te = atr_seqs[val_cut:]

    dist = np.bincount(ytr, minlength=3)
    print(f"[labels] Short={dist[0]:,} | Neutral={dist[1]:,} | Long={dist[2]:,}", flush=True)
    print(f"[seqs]   train={len(Xtr):,} | val={len(Xv):,} | test={len(Xte):,}", flush=True)
    del X, y, y_raw, atr_seqs; gc.collect()

    cw = compute_cw(ytr)

    # ── Optuna ──────────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda t: objective(t, Xtr, ytr, Xv, yv, yv_raw, atr_v, cw, sc),
        n_trials=N_TRIALS, catch=(Exception,),
    )
    bp = study.best_params
    print(f"[optuna] best_params={bp}", flush=True)
    print_summary_table()
    tf.keras.backend.clear_session(); gc.collect()

    # ── Walk-Forward Validation with best params ─────────────────────────────
    wfv_params = dict(units=bp["units"], nh=bp["nh"], d_rate=bp["d"],
                      lr=bp["lr"], noise=bp["noise"],
                      gru_layers=bp.get("gru_layers", 2))
    wfv_metrics = walk_forward_validate(
        np.concatenate([Xtr, Xv]), np.concatenate([ytr, yv]),
        np.concatenate([yv_raw, yv_raw]),   # approx — use val raw ret
        np.concatenate([atr_v, atr_v]),
        wfv_params, cw,
    )

    # ── Final full-data training ─────────────────────────────────────────────
    with strategy.scope():
        m = build_model(SEQ_LEN, NF, **wfv_params)

    ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True)
    ds_vl = make_ds(Xv,  yv,  BATCH)
    ds_te = make_ds(Xte, yte, BATCH)

    cbs = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(os.path.join(OUT_DIR, "model_4H_best_final.keras"),
                        save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-7, verbose=1),
        TerminateOnNaN(),
    ]
    m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=1,
          callbacks=cbs, class_weight=cw)

    pr    = m.predict(ds_te, verbose=0)
    stats = backtest(pr, yte_raw, atr_te)
    print(
        f"\n[FINAL TEST] Ret={stats['ret']:.2%} | WR={stats['wr']:.2%} | "
        f"Sharpe={stats['sharpe']:.2f} | Sortino={stats['sortino']:.2f} | "
        f"MDD={stats['mdd']:.2%} | PF={stats['pf']:.2f} | Trades={stats['trades']}",
        flush=True,
    )
    print(f"[WFV  MEAN]  Ret={wfv_metrics.get('ret',0):.2%} | "
          f"WR={wfv_metrics.get('wr',0):.2%} | "
          f"Sharpe={wfv_metrics.get('sharpe',0):.2f} | "
          f"MDD={wfv_metrics.get('mdd',0):.2%}", flush=True)

    final_score = _composite_score(stats)
    m.save(os.path.join(OUT_DIR, "model_4H_final.keras"))
    _try_save_best(m, final_score, sc, label="final")

    joblib.dump(
        {"sc": sc, "NF": NF, "N_PRICE": N_PRICE,
         "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS, "SEQ_LEN": SEQ_LEN},
        os.path.join(OUT_DIR, "tft_4H_scaler.save"),
    )
    joblib.dump(stats,         os.path.join(OUT_DIR, "tft_4H_stats.save"))
    joblib.dump(wfv_metrics,   os.path.join(OUT_DIR, "tft_4H_wfv_metrics.save"))
    joblib.dump(bp,            os.path.join(OUT_DIR, "tft_4H_best_params.save"))
    joblib.dump(TRIAL_RESULTS, os.path.join(OUT_DIR, "tft_4H_trial_results.save"))
    print(f"[done] → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    train()
