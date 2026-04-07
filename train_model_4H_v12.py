import gc, os, glob, warnings, logging, random
import numpy as np, pandas as pd, tensorflow as tf, optuna, joblib
from numba import njit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Bidirectional, GRU,
    MultiHeadAttention, Add, GlobalAveragePooling1D, GaussianNoise,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau,
)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)
try:
    [tf.config.experimental.set_memory_growth(g, True)
     for g in tf.config.list_physical_devices("GPU")]
except Exception:
    pass
strategy = tf.distribute.MirroredStrategy()
NGPU = max(strategy.num_replicas_in_sync, 1)

BTC_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/BTC"
ETH_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/ETH"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN         = 60
BATCH           = 64
EPOCHS          = 100
N_TRIALS        = 30
EPS             = 1e-8
FEE             = 0.001
TP_MULT         = 2.0
SL_MULT         = 1.0
TB_BARS         = 10
ADX_TREND       = 18.0
ANN_FACTOR      = np.sqrt(365 * 6)
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.10
WF_SPLITS       = 5
MAX_MDD         = -0.25      # kept for reporting only — no hard kill in score
MDD_SOFT        = -0.20
TARGET_WR       = 0.55
TARGET_SHARPE   = 1.50
MIN_EDGE_MULTIPLE = 3.0
SIGNAL_BUFFER   = round(2 * FEE * MIN_EDGE_MULTIPLE, 4)
LOG_RET_PERIODS = [3, 6, 12, 24]
N_CLASSES       = 3
_L2_DENSE       = tf.keras.regularizers.l2(1e-5)
_L2_GRU         = tf.keras.regularizers.l2(1e-3)   # v12: stronger reg on BiGRU
TRIAL_RESULTS   = {}
BEST_SCORE      = -np.inf
FILE_METRICS    = []

# v12: tightened search space
LR_LOW,  LR_HIGH = 5e-5,  1.5e-4   # shield against large LR instability
D_LOW,   D_HIGH  = 0.20,  0.40

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
def triple_barrier_labels(close_arr, atr_arr, adx_arr,
                          tp_m=TP_MULT, sl_m=SL_MULT, nb=TB_BARS):
    n = len(close_arr)
    labels = np.ones(n, dtype=np.int32)
    for i in range(n - nb):
        if adx_arr[i] < ADX_TREND:
            continue
        entry  = close_arr[i]
        tp_lvl = entry * (1 + tp_m * atr_arr[i])
        sl_lvl = entry * (1 - sl_m * atr_arr[i])
        future = close_arr[i+1 : i+1+nb]
        tp_hit = int(np.argmax(future >= tp_lvl)) if (future >= tp_lvl).any() else nb
        sl_hit = int(np.argmax(future <= sl_lvl)) if (future <= sl_lvl).any() else nb
        if tp_hit < sl_hit:
            log_ret_to_tp = np.log(tp_lvl / (entry + EPS))
            if log_ret_to_tp > SIGNAL_BUFFER:
                labels[i] = 2
        elif sl_hit < tp_hit:
            log_ret_to_sl = np.log(entry / (sl_lvl + EPS))
            if log_ret_to_sl > SIGNAL_BUFFER:
                labels[i] = 0
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# BARRIER RETURNS — numba-accelerated
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _compute_barrier_returns_nb(close_arr, atr_arr, sl, n_seqs,
                                 tp_m, sl_m, nb, eps):
    barrier_rets = np.zeros(n_seqs, dtype=np.float32)
    for i in range(n_seqs):
        bar   = sl + i
        entry = close_arr[bar]
        if entry <= 0.0:
            continue
        tp_lvl = entry * (1.0 + tp_m * atr_arr[bar])
        sl_lvl = entry * (1.0 - sl_m * atr_arr[bar])
        tp_hit = nb; sl_hit = nb
        for j in range(1, nb + 1):
            p = close_arr[bar + j]
            if tp_hit == nb and p >= tp_lvl: tp_hit = j - 1
            if sl_hit == nb and p <= sl_lvl: sl_hit = j - 1
            if tp_hit < nb and sl_hit < nb:  break
        exit_idx   = min(tp_hit, sl_hit, nb - 1)
        exit_price = close_arr[bar + 1 + exit_idx]
        barrier_rets[i] = np.log(exit_price / (entry + eps))
    return barrier_rets

def compute_barrier_returns(close_arr, atr_arr, sl=SEQ_LEN, n_seqs=None):
    if n_seqs is None:
        n_seqs = len(close_arr) - sl - TB_BARS
    return _compute_barrier_returns_nb(
        close_arr.astype(np.float64), atr_arr.astype(np.float32),
        sl, n_seqs, float(TP_MULT), float(SL_MULT), int(TB_BARS), EPS
    )


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
    df["macd_norm"]      = macd_r/(c+EPS)
    df["macd_hist_norm"] = (macd_r-sig_r)/(c+EPS)
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
    df["rvi"]       = (v-v.rolling(42).mean())/(v.rolling(42).std()+EPS)
    df["vol_trend"] = v.ewm(span=12,adjust=False).mean()/(v.ewm(span=42,adjust=False).mean()+EPS)
    vpt = (br*v).cumsum()
    df["vpt_norm"]   = vpt/(vpt.rolling(200).std()+EPS)
    df["volatility"] = br.rolling(14).std()
    df["roc"]        = (c-c.shift(14))/(c.shift(14)+EPS)
    df["vol_scaled_ret"] = br/(df["volatility"]+EPS)
    eth_vol = er.rolling(14).std()
    pool = (df["volatility"]+eth_vol)/2
    df["vol_zsc"] = (df["volatility"]-pool.rolling(180).mean())/(pool.rolling(180).std()+EPS)
    obv = pd.Series(np.cumsum(np.sign(br.fillna(0).values)*v.values), index=df.index)
    df["obv_norm"] = obv/(obv.rolling(200).std()+EPS)
    tp_v = (h+l+c)/3
    df["vwap_ratio"]     = c/((tp_v*v).rolling(20).sum()/(v.rolling(20).sum()+EPS)+EPS)
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
    tenkan = (h.rolling(9).max()+l.rolling(9).min())/2
    kijun  = (h.rolling(26).max()+l.rolling(26).min())/2
    s_a = (tenkan+kijun)/2
    s_b = (h.rolling(52).max()+l.rolling(52).min())/2
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
    df["mom_3"]  = (c-c.shift(3))/(c.shift(3)+EPS)
    df["mom_6"]  = (c-c.shift(6))/(c.shift(6)+EPS)
    df["mom_12"] = (c-c.shift(12))/(c.shift(12)+EPS)
    df["mom_24"] = (c-c.shift(24))/(c.shift(24)+EPS)
    df["vol_quantile"] = df["volatility"].rolling(360).rank(pct=True)
    log_c = np.log(c+EPS)
    for p in LOG_RET_PERIODS:
        df[f"log_ret_{p}"] = (log_c-log_c.shift(p)).astype(np.float32)
    d_open = o.resample("1D").first().reindex(df.index,method="ffill").ffill()
    d_atr  = atr_raw.resample("1D").mean().reindex(df.index,method="ffill").ffill()
    d_ema  = c.resample("1D").last().ewm(span=20,adjust=False).mean()\
              .reindex(df.index,method="ffill").ffill()
    df["daily_ret"]            = np.log((c+EPS)/(d_open+EPS)).astype(np.float32)
    df["daily_trend_strength"] = ((c-d_open)/(d_atr+EPS)).clip(-5,5).astype(np.float32)
    df["daily_ema_slope"]      = ((d_ema-d_ema.shift(6))/(d_ema.shift(6)+EPS)).astype(np.float32)
    df["price_vs_daily_open"]  = ((c-d_open)/(c+EPS)).astype(np.float32)
    del d_open, d_atr, d_ema, log_c, up_v, dn_v, obv, vpt
    hr = df.index.hour
    df["hour_sin"] = np.sin(2*np.pi*hr/24).astype(np.float32)
    df["hour_cos"] = np.cos(2*np.pi*hr/24).astype(np.float32)
    df["day_sin"]  = np.sin(2*np.pi*df.index.dayofweek/7).astype(np.float32)
    df["day_cos"]  = np.cos(2*np.pi*df.index.dayofweek/7).astype(np.float32)
    how            = df.index.dayofweek*24+hr
    df["week_sin"] = np.sin(2*np.pi*how/168).astype(np.float32)
    df["week_cos"] = np.cos(2*np.pi*how/168).astype(np.float32)
    df["session_asia"]        = ((hr>=0)&(hr<8)).astype(np.float32)
    df["session_eu"]          = ((hr>=8)&(hr<16)).astype(np.float32)
    df["session_us"]          = ((hr>=16)&(hr<24)).astype(np.float32)
    df["is_trending_session"] = ((hr>=8)&(hr<20)).astype(np.float32)
    adx_raw = adx_v.copy()
    df[PRICE_FEATS] = df[PRICE_FEATS].shift(1)
    df = df.dropna(subset=ALL_FEATS)[ALL_FEATS].astype(np.float32)
    adx_aligned = adx_raw.reindex(df.index).ffill().bfill().values.astype(np.float32)
    return df, adx_aligned


# ══════════════════════════════════════════════════════════════════════════════
# SCALING
# ══════════════════════════════════════════════════════════════════════════════
def _apply_log_transform(feat_arr):
    for idx in _LOG_SCALE_IDX:
        col = feat_arr[:, idx]
        feat_arr[:, idx] = np.sign(col) * np.log1p(np.abs(col)).astype(np.float32)

def fit_scaler(feat_arr, train_end):
    _apply_log_transform(feat_arr)
    sc = RobustScaler()
    feat_arr[:train_end, :N_PRICE] = sc.fit_transform(feat_arr[:train_end, :N_PRICE])
    feat_arr[train_end:, :N_PRICE] = sc.transform(feat_arr[train_end:, :N_PRICE])
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_seqs(feat_arr, close_arr, atr_arr, adx_arr, sl=SEQ_LEN):
    n_seqs = len(feat_arr) - sl - TB_BARS
    idx    = np.arange(n_seqs)[:, None] + np.arange(sl)
    X      = feat_arr[idx]
    labels = triple_barrier_labels(close_arr, atr_arr, adx_arr)
    y      = labels[:n_seqs].astype(np.int32)
    y_raw  = compute_barrier_returns(close_arr, atr_arr, sl, n_seqs)
    atr_s  = atr_arr[sl: sl+n_seqs]
    return X, y, y_raw, atr_s

def _temporal_split(n):
    return int(n*TRAIN_FRAC), int(n*(TRAIN_FRAC+VAL_FRAC))


# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS v12 — Neutral boosted 1.5× to force high-confidence entries
# ══════════════════════════════════════════════════════════════════════════════
def compute_cw(ytr):
    classes = np.array([0, 1, 2])
    raw_w   = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    raw_w[1] *= 1.50          # v12: 1.5× on Neutral (was 0.50 in v11)
    raw_w    /= raw_w.mean()
    cw = {i: float(raw_w[i]) for i in range(3)}
    print(f"[class_weights] Short={cw[0]:.3f} | Neutral={cw[1]:.3f} | Long={cw[2]:.3f}", flush=True)
    return cw


# ══════════════════════════════════════════════════════════════════════════════
# MODEL v12 — l2(1e-3) on BiGRU; units capped at 512
# ══════════════════════════════════════════════════════════════════════════════
def build_model(seq_len, n_feat, units, nh, d_rate, lr, noise, gru_layers=2):
    inp = Input(shape=(seq_len, n_feat), dtype="float32")
    x   = GaussianNoise(noise)(inp)
    x   = Dense(units, kernel_regularizer=_L2_DENSE)(x)
    for _ in range(gru_layers):
        x = Bidirectional(
            GRU(units//2, return_sequences=True,
                dropout=d_rate,
                recurrent_dropout=min(d_rate*0.5, 0.20),
                kernel_regularizer=_L2_GRU,         # v12: BiGRU l2
                recurrent_regularizer=_L2_GRU)      # v12: recurrent l2
        )(x)
        x = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=nh, key_dim=max(units//nh, 1), dropout=d_rate)(x, x)
    x    = LayerNormalization()(Add()([x, attn]))
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(units//2, activation="elu", kernel_regularizer=_L2_DENSE)(x)
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
# BACKTEST v12 — full defensive empty-array guards
# ══════════════════════════════════════════════════════════════════════════════
def backtest(proba, y_raw, atr_arr):
    _default = dict(ret=0.0, mdd=0.0, sharpe=0.0, sortino=0.0,
                    pf=0.0, wr=0.0, trades=0)
    try:
        n     = min(len(proba), len(y_raw), len(atr_arr))
        if n == 0:
            return _default
        proba = proba[:n]; y_raw = y_raw[:n]; atr_ = atr_arr[:n]
        pred  = np.argmax(proba, axis=1)
        conf  = np.max(proba, axis=1)
        raw_sig = np.where(pred==2, 1.0, np.where(pred==0, -1.0, 0.0))
        size    = np.clip((conf - 0.55) / 0.45, 0.0, 1.0)
        sig     = raw_sig * size
        eff     = np.clip(y_raw, -3.0 * atr_, 3.0 * atr_)
        prev_sig   = np.concatenate([[0.0], sig[:-1]])
        trade_flag = (sig != prev_sig).astype(np.float32)
        rets    = sig * eff - 2.0 * FEE * np.abs(sig) * trade_flag
        if len(rets) == 0:
            return _default
        cum = np.cumprod(1.0 + np.clip(rets, -0.99, 10.0))
        pk  = np.maximum.accumulate(cum)
        mdd = float(((cum - pk) / (pk + EPS)).min()) if len(cum) > 0 else 0.0
        ret_std = float(rets.std()) if len(rets) > 1 else EPS
        sh      = float(rets.mean() / (ret_std + EPS) * ANN_FACTOR)
        neg     = rets[rets < 0]
        neg_std = float(neg.std()) if len(neg) > 1 else EPS
        sortino = float(rets.mean() / (neg_std + EPS) * ANN_FACTOR)
        pos_sum = float(rets[rets > 0].sum()) if (rets > 0).any() else 0.0
        neg_sum = float(abs(neg.sum()))        if len(neg) > 0    else EPS
        pf      = pos_sum / (neg_sum + EPS)
        n_tr    = int(np.count_nonzero(trade_flag * (sig != 0)))
        wr      = (float(len(rets[(raw_sig != 0) & (rets > 0)])) /
                   (float(np.count_nonzero(raw_sig)) + EPS))
        total_ret = float(cum[-1] - 1) if len(cum) > 0 else 0.0
        return dict(ret=total_ret, mdd=mdd, sharpe=sh, sortino=sortino,
                    pf=pf, wr=float(wr), trades=n_tr)
    except Exception as e:
        print(f"[backtest] ERROR: {e}", flush=True)
        return _default


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE v12
# Formula: (Sharpe×0.4) + (PF×0.4) + (log(Ret+1)×0.2) - (|MDD|×1.5)
# Soft exponential decay when MDD > 20% — NO hard kill
# ══════════════════════════════════════════════════════════════════════════════
def _composite_score(bt):
    if bt["trades"] == 0:
        return -1.0
    sharpe_s = float(np.clip(bt["sharpe"], -3.0, 10.0))
    pf_s     = float(min(bt["pf"], 10.0))
    ret_s    = float(np.log(max(bt["ret"] + 1.0, 1e-6)))
    mdd_abs  = float(abs(bt["mdd"]))
    score    = (sharpe_s * 0.4) + (pf_s * 0.4) + (ret_s * 0.2) - (mdd_abs * 1.5)
    if mdd_abs > abs(MDD_SOFT):                    # MDD worse than 20%
        score *= float(np.exp(-5.0 * (mdd_abs - abs(MDD_SOFT))))
    return float(score)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL BEST PRESERVATION
# ══════════════════════════════════════════════════════════════════════════════
def _try_save_best(model, score, sc, label=""):
    global BEST_SCORE
    if np.isfinite(score) and score > BEST_SCORE:
        BEST_SCORE = score
        model.save(os.path.join(OUT_DIR, "best_model_v12.keras"))
        joblib.dump(
            {"sc": sc, "NF": NF, "N_PRICE": N_PRICE,
             "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS,
             "SEQ_LEN": SEQ_LEN, "SIGNAL_BUFFER": SIGNAL_BUFFER},
            os.path.join(OUT_DIR, "scaler_v12.pkl"),
        )
        print(f"  ★ [{label}] NEW BEST  score={score:.4f} → best_model_v12.keras", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward_validate(X, y, y_raw, atr_s, params, cw):
    n       = len(X)
    fold_sz = n // (WF_SPLITS + 1)
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
        ds_vl = make_ds(Xv_f,  yv_f,  BATCH)
        cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
              ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=0),
              TerminateOnNaN()]
        mf.fit(ds_tr, validation_data=ds_vl, epochs=60, verbose=0,
               callbacks=cb, class_weight=cw)
        pr  = mf.predict(ds_vl, verbose=0)
        bt  = backtest(pr, yr_f, at_f)
        wf_metrics.append(bt)
        mdd_tag = "🔴" if bt["mdd"] < MAX_MDD else ("⚠️" if bt["mdd"] < MDD_SOFT else "✅")
        print(f"  Fold {fold+1}/{WF_SPLITS} | Ret={bt['ret']:.2%} | "
              f"WR={bt['wr']:.2%} | Sharpe={bt['sharpe']:.2f} | "
              f"MDD={bt['mdd']:.2%} {mdd_tag}", flush=True)
        del mf, ds_tr, ds_vl; gc.collect()
    if not wf_metrics:
        return {}
    agg = {k: float(np.mean([m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    std = {k: float(np.std( [m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    print(f"[WFV] mean → Ret={agg['ret']:.2%} | WR={agg['wr']:.2%} | "
          f"Sharpe={agg['sharpe']:.2f} | MDD={agg['mdd']:.2%}", flush=True)
    agg["_std"] = std
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE v12
# Optimizes on NEGATIVE composite score (direction="minimize")
# Search space: units ∈ [128,256,384,512], lr ∈ [5e-5, 1.5e-4]
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial, Xtr, ytr, Xv, yv, yv_raw, atr_v, cw, sc):
    tf.keras.backend.clear_session(); gc.collect()
    units      = trial.suggest_categorical("units",      [128, 256, 384, 512])  # v12: 768 removed
    nh         = trial.suggest_categorical("nh",         [4, 8, 16])
    d          = trial.suggest_float("d",                D_LOW, D_HIGH)
    lr         = trial.suggest_float("lr",               LR_LOW, LR_HIGH, log=True)  # v12: [5e-5,1.5e-4]
    noise      = trial.suggest_float("noise",            0.002, 0.03)
    gru_layers = trial.suggest_categorical("gru_layers", [2, 3])
    if units % nh != 0:
        return float("inf")
    tn = trial.number + 1
    opt_score = float("inf")
    m = None
    try:
        with strategy.scope():
            m = build_model(SEQ_LEN, NF, units, nh, d, lr, noise, gru_layers)
        ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True)
        ds_vl = make_ds(Xv,  yv,  BATCH)
        cb = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=0),
            TerminateOnNaN(),
        ]
        m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=0,
              callbacks=cb, class_weight=cw)
        pr    = m.predict(ds_vl, verbose=0)
        bt    = backtest(pr, yv_raw, atr_v)
        score = _composite_score(bt)          # v12: Optuna sees composite score, not val_loss
        opt_score = -score                    # minimize → negate
        if np.isfinite(score):
            ckpt = os.path.join(OUT_DIR, f"model_v12_trial_{tn:02d}.keras")
            m.save(ckpt)
            mdd_tag = "🔴" if bt["mdd"] < MAX_MDD else ("⚠️" if bt["mdd"] < MDD_SOFT else "✅")
            # v12: thread-safe dict update via local variable then assign
            result = {
                "score": score, "ret": bt["ret"], "sharpe": bt["sharpe"],
                "mdd": bt["mdd"], "pf": bt["pf"], "wr": bt["wr"],
                "trades": bt["trades"], "model": ckpt,
            }
            TRIAL_RESULTS[tn] = result       # atomic key assignment — safe for single-process Optuna
            print(
                f"[T{tn:02d}/{N_TRIALS}] u={units}|nh={nh}|gl={gru_layers}|"
                f"d={d:.2f}|lr={lr:.1e} | "
                f"Ret={bt['ret']:.2%} | WR={bt['wr']:.2%} | "
                f"Sharpe={bt['sharpe']:.2f} | MDD={bt['mdd']:.2%} | "
                f"PF={bt['pf']:.2f} | score={score:.4f} {mdd_tag}", flush=True,
            )
            _try_save_best(m, score, sc, label=f"T{tn:02d}")
    except Exception as e:
        print(f"[T{tn:02d}] ERROR: {e}", flush=True)
    finally:
        try: del m
        except NameError: pass
        tf.keras.backend.clear_session(); gc.collect()
    return opt_score


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_summary_table():
    if not TRIAL_RESULTS: return
    rows = []
    for tn, r in sorted(TRIAL_RESULTS.items()):
        mdd_tag = "🔴" if r.get("mdd", 0) < MAX_MDD else ("⚠️" if r.get("mdd", 0) < MDD_SOFT else "✅")
        wr_ok   = "✅" if r.get("wr", 0)     >= TARGET_WR     else "  "
        sh_ok   = "✅" if r.get("sharpe", 0) >= TARGET_SHARPE else "  "
        rows.append({
            "T":      tn,
            "Ret":    f"{r.get('ret', 0):.2%}",
            "WR":     f"{r.get('wr', 0):.2%}{wr_ok}",
            "Sharpe": f"{r.get('sharpe', 0):.2f}{sh_ok}",
            "MDD":    f"{r.get('mdd', 0):.2%}{mdd_tag}",
            "PF":     f"{r.get('pf', 0):.2f}",
            "Trades": r.get("trades", 0),
            "Score":  f"{r.get('score', -99):.4f}",
        })
    df = pd.DataFrame(rows).set_index("T")
    valid   = {t: r for t, r in TRIAL_RESULTS.items() if r.get("mdd", -1) >= MAX_MDD}
    best_tn = max(valid, key=lambda t: valid[t].get("score", -np.inf)) if valid else None
    print("\n" + "="*95)
    print("TRIAL COMPARISON SUMMARY  v12  (🔴=MDD>25% | ⚠️=MDD>20% | ✅=OK)")
    print("="*95)
    print(df.to_string())
    print("="*95)
    if best_tn:
        b = TRIAL_RESULTS[best_tn]
        print(f"★ BEST: T{best_tn:02d} | Score={b.get('score',0):.4f} | "
              f"Ret={b.get('ret',0):.2%} | WR={b.get('wr',0):.2%} | "
              f"Sharpe={b.get('sharpe',0):.2f} | MDD={b.get('mdd',0):.2%} | "
              f"PF={b.get('pf',0):.2f}")
    print("="*95 + "\n", flush=True)
    joblib.dump(df, os.path.join(OUT_DIR, "trial_summary_v12.save"))


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def print_stability_report(wfv_metrics):
    print("\n" + "="*90)
    print("STABILITY REPORT  v12")
    print("="*90)
    if FILE_METRICS:
        keys = ["ret", "sharpe", "mdd", "wr", "pf"]
        print(f"\n{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CV%':>8}")
        print("-"*56)
        for k in keys:
            vals = np.array([m[k] for m in FILE_METRICS if k in m])
            if len(vals) == 0: continue
            mean_v = vals.mean(); std_v = vals.std()
            cv     = abs(std_v / (mean_v + EPS)) * 100
            print(f"{k:<12} {mean_v:>10.4f} {std_v:>10.4f} "
                  f"{vals.min():>10.4f} {vals.max():>10.4f} {cv:>7.1f}%")
        print()
    if TRIAL_RESULTS:
        valid = [r for r in TRIAL_RESULTS.values() if r.get("mdd", -1) >= MAX_MDD]
        if valid:
            for k in ["ret", "sharpe", "mdd", "wr"]:
                vals = np.array([r.get(k, 0) for r in valid])
                print(f"[Trials|{k}]  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                      f"min={vals.min():.4f}  max={vals.max():.4f}")
    if wfv_metrics and "_std" in wfv_metrics:
        std = wfv_metrics["_std"]
        print(f"\n[WFV fold std] Ret={std.get('ret',0):.2%} | WR={std.get('wr',0):.2%} | "
              f"Sharpe={std.get('sharpe',0):.2f} | MDD={std.get('mdd',0):.2%}")
        sharpe_cv = abs(std.get('sharpe',0) / (wfv_metrics.get('sharpe',EPS) + EPS)) * 100
        print(f"[WFV Sharpe CV] {sharpe_cv:.1f}%  "
              f"{'✅ Consistent' if sharpe_cv < 40 else '⚠️  High variance'}")
    print("="*90 + "\n", flush=True)
    joblib.dump(
        {"file_metrics": FILE_METRICS, "trial_results": TRIAL_RESULTS, "wfv": wfv_metrics},
        os.path.join(OUT_DIR, "stability_report_v12.save"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAIN LOOP v12
# ══════════════════════════════════════════════════════════════════════════════
def train():
    print(f"[GRU-4H-v12] GPUs={NGPU} | SEQ={SEQ_LEN} | NF={NF} | "
          f"FEE={FEE*100:.1f}%×2 | BUF={SIGNAL_BUFFER*100:.2f}% | "
          f"MDD_HARD={MAX_MDD:.0%} | MDD_SOFT={MDD_SOFT:.0%} | ADX≥{ADX_TREND} | "
          f"TRIALS={N_TRIALS} | WF_SPLITS={WF_SPLITS} | "
          f"LR=[{LR_LOW:.0e},{LR_HIGH:.0e}] | D=[{D_LOW},{D_HIGH}] | "
          f"units_max=512 | Optuna→CompositeScore", flush=True)

    btc_raw = load_sync_resample(BTC_DIR, "BTC")
    eth_raw = load_sync_resample(ETH_DIR, "ETH")
    btc_raw, eth_raw = sync_assets(btc_raw, eth_raw); gc.collect()

    c_r  = btc_raw["close"]; h_r = btc_raw["high"]; l_r = btc_raw["low"]
    tr_r = pd.concat([(h_r-l_r),(h_r-c_r.shift()).abs(),(l_r-c_r.shift()).abs()],axis=1).max(axis=1)
    atr_norm_u = (tr_r.ewm(com=13,adjust=False).mean()/(c_r+EPS)).astype(np.float32)

    feat_df, adx_aligned = mkfeat(btc_raw, eth_raw)
    close_s = btc_raw["close"].loc[feat_df.index]
    atr_s   = atr_norm_u.reindex(feat_df.index).ffill().bfill()
    del btc_raw, eth_raw, tr_r, c_r, h_r, l_r, atr_norm_u; gc.collect()

    n_seqs = len(feat_df) - SEQ_LEN - TB_BARS
    tr_cut, val_cut = _temporal_split(n_seqs)
    feat_arr = feat_df.values.astype(np.float32)
    del feat_df; gc.collect()

    sc = fit_scaler(feat_arr, tr_cut+SEQ_LEN); gc.collect()
    c_arr   = close_s.values.astype(np.float64)
    atr_arr = atr_s.values.astype(np.float32)
    del close_s, atr_s; gc.collect()

    X, y, y_raw, atr_seqs = build_seqs(feat_arr, c_arr, atr_arr, adx_aligned)
    del feat_arr, c_arr, atr_arr, adx_aligned; gc.collect()

    Xtr = X[:tr_cut];          ytr = y[:tr_cut]
    Xv  = X[tr_cut:val_cut];   yv  = y[tr_cut:val_cut]
    Xte = X[val_cut:];         yte = y[val_cut:]
    yv_raw  = y_raw[tr_cut:val_cut];  yte_raw = y_raw[val_cut:]
    atr_v   = atr_seqs[tr_cut:val_cut]; atr_te = atr_seqs[val_cut:]

    dist = np.bincount(ytr, minlength=3)
    print(f"[labels] Short={dist[0]:,} | Neutral={dist[1]:,} | Long={dist[2]:,}", flush=True)
    print(f"[seqs]   train={len(Xtr):,} | val={len(Xv):,} | test={len(Xte):,}", flush=True)
    del X, y, y_raw, atr_seqs; gc.collect()

    cw = compute_cw(ytr)

    study = optuna.create_study(
        direction="minimize",                          # minimizing -composite_score
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda t: objective(t, Xtr, ytr, Xv, yv, yv_raw, atr_v, cw, sc),
        n_trials=N_TRIALS, catch=(Exception,),
    )
    bp = study.best_params
    print(f"\n[optuna] best_params={bp}", flush=True)
    print_summary_table()
    tf.keras.backend.clear_session(); gc.collect()

    wfv_params = dict(units=bp["units"], nh=bp["nh"], d_rate=bp["d"],
                      lr=bp["lr"], noise=bp["noise"],
                      gru_layers=bp.get("gru_layers", 2))
    wfv_metrics = walk_forward_validate(
        np.concatenate([Xtr, Xv]),
        np.concatenate([ytr, yv]),
        np.concatenate([yv_raw, yte_raw[:len(yv_raw)]]),
        np.concatenate([atr_v,  atr_te[:len(atr_v)]]),
        wfv_params, cw,
    )

    with strategy.scope():
        m = build_model(SEQ_LEN, NF, **wfv_params)

    ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True)
    ds_vl = make_ds(Xv,  yv,  BATCH)
    ds_te = make_ds(Xte, yte, BATCH)

    cbs = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(os.path.join(OUT_DIR, "model_v12_best_final.keras"),
                        save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-7, verbose=1),
        TerminateOnNaN(),
    ]
    m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=1,
          callbacks=cbs, class_weight=cw)

    pr    = m.predict(ds_te, verbose=0)
    stats = backtest(pr, yte_raw, atr_te)
    FILE_METRICS.append(stats)
    print(
        f"\n[FINAL TEST] Ret={stats['ret']:.2%} | WR={stats['wr']:.2%} | "
        f"Sharpe={stats['sharpe']:.2f} | Sortino={stats['sortino']:.2f} | "
        f"MDD={stats['mdd']:.2%} | PF={stats['pf']:.2f} | Trades={stats['trades']}",
        flush=True,
    )
    if wfv_metrics:
        print(f"[WFV  MEAN]  Ret={wfv_metrics.get('ret',0):.2%} | "
              f"WR={wfv_metrics.get('wr',0):.2%} | "
              f"Sharpe={wfv_metrics.get('sharpe',0):.2f} | "
              f"MDD={wfv_metrics.get('mdd',0):.2%}", flush=True)

    print_stability_report(wfv_metrics)

    final_score = _composite_score(stats)
    m.save(os.path.join(OUT_DIR, "model_v12_final.keras"))
    _try_save_best(m, final_score, sc, label="final")

    joblib.dump(
        {"sc": sc, "NF": NF, "N_PRICE": N_PRICE,
         "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS,
         "SEQ_LEN": SEQ_LEN, "SIGNAL_BUFFER": SIGNAL_BUFFER},
        os.path.join(OUT_DIR, "scaler_v12.pkl"),
    )
    joblib.dump(stats,         os.path.join(OUT_DIR, "stats_v12.save"))
    joblib.dump(wfv_metrics,   os.path.join(OUT_DIR, "wfv_metrics_v12.save"))
    joblib.dump(bp,            os.path.join(OUT_DIR, "best_params_v12.save"))
    joblib.dump(TRIAL_RESULTS, os.path.join(OUT_DIR, "trial_results_v12.save"))
    print(f"[done] → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    train()
