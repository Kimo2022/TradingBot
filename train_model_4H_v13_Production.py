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

SEQ_LEN           = 60
BATCH             = 64
EPOCHS            = 100
N_TRIALS          = 30
EPS               = 1e-8
FEE               = 0.001
TP_MULT           = 2.0
SL_MULT           = 1.0
TB_BARS           = 10
ADX_TREND         = 18.0
ANN_FACTOR        = np.sqrt(365 * 6)
TRAIN_FRAC        = 0.70
VAL_FRAC          = 0.10
WF_SPLITS         = 5
MAX_MDD           = -0.25
MDD_SOFT          = -0.18
MDD_LIN_THRESH    = -0.10
TARGET_WR         = 0.55
TARGET_SORTINO    = 1.80
MIN_EDGE_MULTIPLE = 3.0
SIGNAL_BUFFER     = round(2 * FEE * MIN_EDGE_MULTIPLE, 4)
LOG_RET_PERIODS   = [3, 6, 12, 24]
N_CLASSES         = 3
_L2_DENSE         = tf.keras.regularizers.l2(1e-5)
_L2_GRU           = tf.keras.regularizers.l2(1e-3)
TRIAL_RESULTS     = {}
BEST_SCORE        = -np.inf
FILE_METRICS      = []

LR_LOW,  LR_HIGH = 5e-5, 1.5e-4
D_LOW,   D_HIGH  = 0.20, 0.40
NW_LOW,  NW_HIGH = 1.0,  2.5

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
            if np.log(tp_lvl / (entry + EPS)) > SIGNAL_BUFFER:
                labels[i] = 2
        elif sl_hit < tp_hit:
            if np.log(entry / (sl_lvl + EPS)) > SIGNAL_BUFFER:
                labels[i] = 0
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# BARRIER RETURNS
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
    ema20  = c.ewm(span=20, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200,adjust=False).mean()
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
    df["body_norm"]        = (c-o).abs()/(atr_raw+EPS)
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
    df["vpt_norm"]       = vpt/(vpt.rolling(200).std()+EPS)
    df["volatility"]     = br.rolling(14).std()
    df["roc"]            = (c-c.shift(14))/(c.shift(14)+EPS)
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
    d_ema  = (c.resample("1D").last().ewm(span=20,adjust=False).mean()
              .reindex(df.index,method="ffill").ffill())
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

def fit_scaler(feat_arr_log, train_end_raw):
    sc = RobustScaler()
    feat_arr_log[:train_end_raw, :N_PRICE] = sc.fit_transform(feat_arr_log[:train_end_raw, :N_PRICE])
    feat_arr_log[train_end_raw:, :N_PRICE] = sc.transform(feat_arr_log[train_end_raw:, :N_PRICE])
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
    atr_s  = atr_arr[sl : sl+n_seqs]
    return X, y, y_raw, atr_s

def _temporal_split(n):
    return int(n*TRAIN_FRAC), int(n*(TRAIN_FRAC+VAL_FRAC))


# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_cw(ytr, neutral_weight=1.5):
    classes = np.array([0, 1, 2])
    raw_w   = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    raw_w[1] *= float(neutral_weight)
    raw_w    /= raw_w.mean()
    cw = {i: float(raw_w[i]) for i in range(3)}
    print(f"[cw nw={neutral_weight:.2f}] Short={cw[0]:.3f} | Neutral={cw[1]:.3f} | Long={cw[2]:.3f}",
          flush=True)
    return cw


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model(seq_len, n_feat, units, nh, d_rate, lr, noise, gru_layers=2):
    inp = Input(shape=(seq_len, n_feat), dtype="float32")
    x   = GaussianNoise(noise)(inp)
    x   = Dense(units, kernel_regularizer=_L2_DENSE)(x)
    for _ in range(gru_layers):
        x = Bidirectional(
            GRU(units//2, return_sequences=True,
                dropout=d_rate,
                recurrent_dropout=max(d_rate * 0.3, 0.10),
                kernel_regularizer=_L2_GRU,
                recurrent_regularizer=_L2_GRU)
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

def make_ds_eval(X, batch):
    return (tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
            .batch(batch * NGPU, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE))


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def backtest(proba, y_raw, atr_arr):
    _default = dict(ret=0.0, mdd=0.0, sharpe=0.0, sortino=0.0,
                    pf=0.0, wr=0.0, trades=0, top_ratio=0.0)
    try:
        n     = min(len(proba), len(y_raw), len(atr_arr))
        if n == 0: return _default
        proba = proba[:n]; y_raw = y_raw[:n]; atr_ = atr_arr[:n]
        pred    = np.argmax(proba, axis=1)
        conf    = np.max(proba,  axis=1)
        raw_sig = np.where(pred==2, 1.0, np.where(pred==0, -1.0, 0.0))
        size    = np.clip((conf - 0.55) / 0.45, 0.0, 1.0)
        sig     = raw_sig * size
        eff     = np.clip(y_raw, -3.0 * atr_, 3.0 * atr_)
        prev_sig   = np.concatenate([[0.0], sig[:-1]])
        trade_flag = (sig != prev_sig).astype(np.float32)
        rets    = sig * eff - 2.0 * FEE * np.abs(sig) * trade_flag
        if len(rets) == 0: return _default
        cum = np.cumprod(1.0 + np.clip(rets, -0.99, 10.0))
        pk  = np.maximum.accumulate(cum)
        mdd = float(((cum - pk) / (pk + EPS)).min())
        ret_std = float(rets.std()) if len(rets) > 1 else EPS
        sh      = float(rets.mean() / (ret_std + EPS) * ANN_FACTOR)
        neg     = rets[rets < 0]
        neg_std = float(neg.std()) if len(neg) > 1 else EPS
        sortino = float(rets.mean() / (neg_std + EPS) * ANN_FACTOR)
        pos_sum = float(rets[rets > 0].sum()) if (rets > 0).any() else 0.0
        neg_sum = float(abs(neg.sum()))        if len(neg) > 0    else EPS
        pf      = pos_sum / (neg_sum + EPS)
        n_tr    = int(np.count_nonzero(trade_flag * (sig != 0)))
        active  = sig != 0.0
        n_act   = int(np.count_nonzero(active))
        wr      = float(np.sum(active & (rets > 0))) / max(n_act, 1)
        pos_rets = rets[active & (rets > 0)]
        top_ratio = float(pos_rets.max() / (pos_rets.sum() + EPS)) if len(pos_rets) > 0 else 0.0
        return dict(ret=float(cum[-1]-1), mdd=mdd, sharpe=sh, sortino=sortino,
                    pf=pf, wr=wr, trades=n_tr, top_ratio=top_ratio)
    except Exception as e:
        print(f"[backtest] ERROR: {e}", flush=True)
        return _default


# ══════════════════════════════════════════════════════════════════════════════
# FOLD SCORE — single-fold base score (input to stability formula)
# ══════════════════════════════════════════════════════════════════════════════
def _fold_score(bt):
    if bt["trades"] == 0:
        return -2.0
    sortino_s = float(np.clip(bt["sortino"], -3.0, 15.0))
    pf_s      = float(min(bt["pf"], 10.0))
    ret_s     = float(np.log(max(bt["ret"] + 1.0, 1e-6)))
    mdd_abs   = float(abs(bt["mdd"]))
    lin_t  = abs(MDD_LIN_THRESH)
    quad_t = abs(MDD_SOFT)
    if mdd_abs <= lin_t:
        mdd_pen = mdd_abs * 0.5
    elif mdd_abs <= quad_t:
        mdd_pen = lin_t * 0.5 + (mdd_abs - lin_t)**2 * 10.0
    else:
        mdd_pen = (lin_t * 0.5
                   + (quad_t - lin_t)**2 * 10.0
                   + (mdd_abs - quad_t) * 25.0)
    return float((sortino_s * 0.40) + (pf_s * 0.35) + (ret_s * 0.25) - mdd_pen)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE — Production Master Logic
# ══════════════════════════════════════════════════════════════════════════════
def _composite_score_v13(bt, wfv_scores=None, val_loss=0.0):
    if bt["trades"] == 0:
        return -2.0

    if wfv_scores and len(wfv_scores) > 0:
        arr    = np.array(wfv_scores, dtype=np.float64)
        mean_s = float(np.mean(arr))
        min_s  = float(np.min(arr))
        std_s  = float(np.std(arr))
        score  = mean_s * 0.7 + min_s * 0.3 - std_s * 0.5
        score += 0.3 * min_s
        if min_s < -0.15:
            score -= 1.5
    else:
        score = _fold_score(bt)

    top_ratio = bt.get("top_ratio", 0.0)
    if top_ratio > 0.4:
        score -= (top_ratio - 0.4) * 2.0

    n_tr = bt["trades"]
    if n_tr < 20:
        score *= 0.5
    elif n_tr > 1500:
        score *= 0.7

    score -= 0.1 * float(val_loss)

    return float(score)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL BEST PRESERVATION
# ══════════════════════════════════════════════════════════════════════════════
def _try_save_best(model, score, sc, label=""):
    global BEST_SCORE
    if np.isfinite(score) and score > BEST_SCORE:
        BEST_SCORE = score
        model.save(os.path.join(OUT_DIR, "best_model_v13.keras"))
        joblib.dump(
            {"sc": sc, "NF": NF, "N_PRICE": N_PRICE,
             "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS,
             "SEQ_LEN": SEQ_LEN, "SIGNAL_BUFFER": SIGNAL_BUFFER},
            os.path.join(OUT_DIR, "scaler_v13.pkl"),
        )
        print(f"  ★ [{label}] NEW BEST  score={score:.4f} → best_model_v13.keras", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward_validate(feat_arr_log, c_arr, atr_arr, adx_arr,
                           n_wfv, params, cw):
    fold_sz    = n_wfv // (WF_SPLITS + 1)
    wf_metrics = []
    fold_scores = []
    print(f"\n[WFV-v13] {WF_SPLITS} folds | fold_size≈{fold_sz:,} | per-fold scaler ✓", flush=True)

    for fold in range(WF_SPLITS):
        train_end_seq = fold_sz * (fold + 1)
        val_end_seq   = min(train_end_seq + fold_sz, n_wfv)
        n_val         = val_end_seq - train_end_seq

        if n_val < 200:
            print(f"  Fold {fold+1}: only {n_val} val seqs — skipping.", flush=True)
            break

        train_end_raw = train_end_seq + SEQ_LEN
        total_raw     = min(val_end_seq + SEQ_LEN + TB_BARS, len(feat_arr_log))

        assert train_end_raw <= total_raw, \
            f"Fold {fold+1}: train_end_raw {train_end_raw} > total_raw {total_raw}"
        assert total_raw <= len(feat_arr_log), \
            f"Fold {fold+1}: total_raw {total_raw} exceeds feat_arr_log len {len(feat_arr_log)}"
        assert total_raw <= len(c_arr), \
            f"Fold {fold+1}: total_raw {total_raw} exceeds c_arr len {len(c_arr)}"
        assert total_raw <= len(atr_arr), \
            f"Fold {fold+1}: total_raw {total_raw} exceeds atr_arr len {len(atr_arr)}"

        fold_feat = feat_arr_log[:total_raw].copy()
        sc_fold   = RobustScaler()
        fold_feat[:train_end_raw, :N_PRICE] = sc_fold.fit_transform(
            fold_feat[:train_end_raw, :N_PRICE])
        fold_feat[train_end_raw:, :N_PRICE] = sc_fold.transform(
            fold_feat[train_end_raw:, :N_PRICE])

        fold_c   = c_arr[:total_raw]
        fold_atr = atr_arr[:total_raw]

        assert len(fold_feat) == total_raw, \
            f"Fold {fold+1}: fold_feat len {len(fold_feat)} != total_raw {total_raw}"
        assert len(fold_c) == total_raw, \
            f"Fold {fold+1}: fold_c len {len(fold_c)} != total_raw {total_raw}"
        assert len(fold_atr) == total_raw, \
            f"Fold {fold+1}: fold_atr len {len(fold_atr)} != total_raw {total_raw}"

        if len(adx_arr) >= total_raw:
            fold_adx = adx_arr[:total_raw]
        else:
            fold_adx = np.concatenate(
                [adx_arr, np.full(total_raw - len(adx_arr), adx_arr[-1], dtype=np.float32)])

        assert len(fold_adx) == total_raw, \
            f"Fold {fold+1}: fold_adx len {len(fold_adx)} != total_raw {total_raw}"

        X_f, y_f, yr_f, at_f = build_seqs(fold_feat, fold_c, fold_atr, fold_adx)
        del fold_feat; gc.collect()

        assert len(X_f)  >= val_end_seq, \
            f"Fold {fold+1}: built {len(X_f)} seqs < val_end {val_end_seq}"
        assert len(yr_f) >= val_end_seq, \
            f"Fold {fold+1}: yr_f {len(yr_f)} < val_end {val_end_seq}"
        assert len(at_f) >= val_end_seq, \
            f"Fold {fold+1}: at_f {len(at_f)} < val_end {val_end_seq}"

        Xtr_f = X_f[:train_end_seq];              ytr_f = y_f[:train_end_seq]
        Xv_f  = X_f[train_end_seq:val_end_seq];   yv_f  = y_f[train_end_seq:val_end_seq]
        yr_vf = yr_f[train_end_seq:val_end_seq]
        at_vf = at_f[train_end_seq:val_end_seq]

        assert len(Xv_f)  == n_val, f"Fold {fold+1}: Xv_f {len(Xv_f)} != n_val {n_val}"
        assert len(yr_vf) == n_val, f"Fold {fold+1}: yr_vf {len(yr_vf)} != n_val {n_val}"
        assert len(at_vf) == n_val, f"Fold {fold+1}: at_vf {len(at_vf)} != n_val {n_val}"

        del X_f, y_f, yr_f, at_f; gc.collect()

        tf.keras.backend.clear_session(); gc.collect()
        with strategy.scope():
            mf = build_model(SEQ_LEN, NF, **params)

        ds_tr = make_ds(Xtr_f, ytr_f, BATCH, shuffle=True)
        ds_vl = make_ds(Xv_f,  yv_f,  BATCH)
        cb = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                              min_lr=1e-7, verbose=0),
            TerminateOnNaN(),
        ]
        hist = mf.fit(ds_tr, validation_data=ds_vl, epochs=60, verbose=0,
                      callbacks=cb, class_weight=cw)

        fold_val_loss = float(min(hist.history.get("val_loss", [1e9])))

        pr        = mf.predict(make_ds_eval(Xv_f, BATCH), verbose=0)
        n_aligned = min(len(pr), n_val)
        bt        = backtest(pr[:n_aligned], yr_vf[:n_aligned], at_vf[:n_aligned])
        wf_metrics.append(bt)

        fs = _fold_score(bt)
        fold_scores.append(fs)

        mdd_tag = "🔴" if bt["mdd"] < MAX_MDD else ("⚠️" if bt["mdd"] < MDD_SOFT else "✅")
        print(f"  Fold {fold+1}/{WF_SPLITS} | Ret={bt['ret']:.2%} | WR={bt['wr']:.2%} | "
              f"Sharpe={bt['sharpe']:.2f} | Sortino={bt['sortino']:.2f} | "
              f"MDD={bt['mdd']:.2%} {mdd_tag} | FoldScore={fs:.4f} | VL={fold_val_loss:.4f}",
              flush=True)
        del mf, Xtr_f, ytr_f, Xv_f, yv_f, yr_vf, at_vf; gc.collect()

    if not wf_metrics:
        return {}
    agg = {k: float(np.mean([m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    std = {k: float(np.std( [m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    print(f"[WFV] mean → Ret={agg['ret']:.2%} | WR={agg['wr']:.2%} | "
          f"Sharpe={agg['sharpe']:.2f} | Sortino={agg['sortino']:.2f} | "
          f"MDD={agg['mdd']:.2%} | FoldScores={[f'{s:.3f}' for s in fold_scores]}",
          flush=True)
    agg["_std"]         = std
    agg["_fold_scores"] = fold_scores
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial, Xtr, ytr, Xv, yv, yv_raw, atr_v, sc):
    tf.keras.backend.clear_session(); gc.collect()
    units          = trial.suggest_categorical("units",        [128, 256, 384, 512])
    nh             = trial.suggest_categorical("nh",           [4, 8, 16])
    d              = trial.suggest_float("d",                  D_LOW, D_HIGH)
    lr             = trial.suggest_float("lr",                 LR_LOW, LR_HIGH, log=True)
    noise          = trial.suggest_float("noise",              0.002, 0.03)
    gru_layers     = trial.suggest_categorical("gru_layers",   [2, 3])
    neutral_weight = trial.suggest_float("neutral_weight",     NW_LOW, NW_HIGH)

    if units % nh != 0:
        return float("inf")

    tn        = trial.number + 1
    opt_score = float("inf")
    m         = None
    try:
        cw_trial = compute_cw(ytr, neutral_weight=neutral_weight)
        with strategy.scope():
            m = build_model(SEQ_LEN, NF, units, nh, d, lr, noise, gru_layers)

        ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True)
        ds_vl = make_ds(Xv,  yv,  BATCH)

        class _PruneCallback(tf.keras.callbacks.Callback):
            def __init__(self, trial, Xv, yv_raw, atr_v):
                super().__init__()
                self._trial       = trial
                self._Xv          = Xv
                self._yr          = yv_raw
                self._at          = atr_v
                self.last_val_loss = 1e9

            def on_epoch_end(self, epoch, logs=None):
                self.last_val_loss = float(logs.get("val_loss", 1e9))
                pr = self.model.predict(make_ds_eval(self._Xv, BATCH), verbose=0)
                n  = min(len(pr), len(self._yr), len(self._at))
                bt = backtest(pr[:n], self._yr[:n], self._at[:n])
                s  = _fold_score(bt) - 0.1 * self.last_val_loss
                self._trial.report(-s, epoch)
                if self._trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        prune_cb = _PruneCallback(trial, Xv, yv_raw, atr_v)

        cb = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                              min_lr=1e-7, verbose=0),
            TerminateOnNaN(),
            prune_cb,
        ]
        m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=0,
              callbacks=cb, class_weight=cw_trial)

        final_val_loss = prune_cb.last_val_loss
        pr    = m.predict(make_ds_eval(Xv, BATCH), verbose=0)
        n_min = min(len(pr), len(yv_raw), len(atr_v))
        bt    = backtest(pr[:n_min], yv_raw[:n_min], atr_v[:n_min])
        score = _composite_score_v13(bt, wfv_scores=None, val_loss=final_val_loss)
        opt_score = -score

        if np.isfinite(score):
            ckpt = os.path.join(OUT_DIR, f"model_v13_trial_{tn:02d}.keras")
            m.save(ckpt)
            mdd_tag = "🔴" if bt["mdd"] < MAX_MDD else ("⚠️" if bt["mdd"] < MDD_SOFT else "✅")
            TRIAL_RESULTS[tn] = {
                "score": score, "ret": bt["ret"], "sharpe": bt["sharpe"],
                "sortino": bt["sortino"], "mdd": bt["mdd"], "pf": bt["pf"],
                "wr": bt["wr"], "trades": bt["trades"], "top_ratio": bt["top_ratio"],
                "neutral_weight": neutral_weight, "model": ckpt,
                "val_loss": final_val_loss,
            }
            print(
                f"[T{tn:02d}/{N_TRIALS}] u={units}|nh={nh}|gl={gru_layers}|"
                f"d={d:.2f}|lr={lr:.1e}|nw={neutral_weight:.2f} | "
                f"Ret={bt['ret']:.2%}|WR={bt['wr']:.2%}|"
                f"Sortino={bt['sortino']:.2f}|MDD={bt['mdd']:.2%}|"
                f"TR={bt['top_ratio']:.3f}|score={score:.4f} {mdd_tag}", flush=True,
            )
            _try_save_best(m, score, sc, label=f"T{tn:02d}")

    except optuna.exceptions.TrialPruned:
        print(f"[T{tn:02d}] PRUNED", flush=True)
        opt_score = float("inf")
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
        mdd_tag = "🔴" if r.get("mdd",0) < MAX_MDD else ("⚠️" if r.get("mdd",0) < MDD_SOFT else "✅")
        wr_ok   = "✅" if r.get("wr",0)      >= TARGET_WR      else "  "
        so_ok   = "✅" if r.get("sortino",0) >= TARGET_SORTINO else "  "
        rows.append({
            "T":        tn,
            "Ret":      f"{r.get('ret',0):.2%}",
            "WR":       f"{r.get('wr',0):.2%}{wr_ok}",
            "Sortino":  f"{r.get('sortino',0):.2f}{so_ok}",
            "Sharpe":   f"{r.get('sharpe',0):.2f}",
            "MDD":      f"{r.get('mdd',0):.2%}{mdd_tag}",
            "PF":       f"{r.get('pf',0):.2f}",
            "TopRatio": f"{r.get('top_ratio',0):.3f}",
            "NW":       f"{r.get('neutral_weight',0):.2f}",
            "VL":       f"{r.get('val_loss',0):.4f}",
            "Trades":   r.get("trades",0),
            "Score":    f"{r.get('score',-99):.4f}",
        })
    df = pd.DataFrame(rows).set_index("T")
    valid   = {t: r for t, r in TRIAL_RESULTS.items() if r.get("mdd",-1) >= MAX_MDD}
    best_tn = max(valid, key=lambda t: valid[t].get("score",-np.inf)) if valid else None
    print("\n" + "="*115)
    print("TRIAL SUMMARY  v13-Production  (🔴=MDD>25% | ⚠️=MDD>18% | ✅=OK)")
    print("="*115)
    print(df.to_string())
    print("="*115)
    if best_tn:
        b = TRIAL_RESULTS[best_tn]
        print(f"★ BEST: T{best_tn:02d} | Score={b.get('score',0):.4f} | "
              f"Ret={b.get('ret',0):.2%} | WR={b.get('wr',0):.2%} | "
              f"Sortino={b.get('sortino',0):.2f} | MDD={b.get('mdd',0):.2%} | "
              f"NW={b.get('neutral_weight',0):.2f} | TR={b.get('top_ratio',0):.3f}")
    print("="*115 + "\n", flush=True)
    joblib.dump(df, os.path.join(OUT_DIR, "trial_summary_v13.save"))


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def print_stability_report(wfv_metrics):
    print("\n" + "="*90)
    print("STABILITY REPORT  v13-Production")
    print("="*90)
    if FILE_METRICS:
        keys = ["ret","sharpe","sortino","mdd","wr","pf","top_ratio"]
        print(f"\n{'Metric':<14} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CV%':>8}")
        print("-"*62)
        for k in keys:
            vals = np.array([m[k] for m in FILE_METRICS if k in m])
            if len(vals) == 0: continue
            mean_v = vals.mean(); std_v = vals.std()
            cv     = abs(std_v / (mean_v + EPS)) * 100
            print(f"{k:<14} {mean_v:>10.4f} {std_v:>10.4f} "
                  f"{vals.min():>10.4f} {vals.max():>10.4f} {cv:>7.1f}%")
        print()
    if TRIAL_RESULTS:
        valid = [r for r in TRIAL_RESULTS.values() if r.get("mdd",-1) >= MAX_MDD]
        if valid:
            for k in ["ret","sharpe","sortino","mdd","wr"]:
                vals = np.array([r.get(k,0) for r in valid])
                print(f"[Trials|{k}]  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                      f"min={vals.min():.4f}  max={vals.max():.4f}")
    if wfv_metrics and "_std" in wfv_metrics:
        std = wfv_metrics["_std"]
        print(f"\n[WFV fold std] Ret={std.get('ret',0):.2%} | WR={std.get('wr',0):.2%} | "
              f"Sortino={std.get('sortino',0):.2f} | MDD={std.get('mdd',0):.2%}")
        sortino_ref = abs(wfv_metrics.get("sortino", EPS))
        sortino_cv  = abs(std.get("sortino",0)) / (sortino_ref + EPS) * 100
        print(f"[WFV Sortino CV] {sortino_cv:.1f}%  "
              f"{'✅ Consistent' if sortino_cv < 40 else '⚠️  High variance'}")
        fold_scores = wfv_metrics.get("_fold_scores", [])
        if fold_scores:
            arr = np.array(fold_scores)
            print(f"[WFV FoldScores] {[f'{s:.4f}' for s in fold_scores]}")
            print(f"[WFV Stability]  mean={arr.mean():.4f} | min={arr.min():.4f} | "
                  f"std={arr.std():.4f} | "
                  f"stability={(arr.mean()*0.7 + arr.min()*0.3 - arr.std()*0.5):.4f}")
    print("="*90 + "\n", flush=True)
    joblib.dump(
        {"file_metrics": FILE_METRICS, "trial_results": TRIAL_RESULTS, "wfv": wfv_metrics},
        os.path.join(OUT_DIR, "stability_report_v13.save"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAIN LOOP — v13 Production
# ══════════════════════════════════════════════════════════════════════════════
def train():
    print(f"[BiGRU-4H-v13-Production] GPUs={NGPU} | SEQ={SEQ_LEN} | NF={NF} | "
          f"FEE={FEE*100:.1f}%×2 | BUF={SIGNAL_BUFFER*100:.2f}% | "
          f"MDD_HARD={MAX_MDD:.0%} | MDD_SOFT={MDD_SOFT:.0%} | ADX≥{ADX_TREND} | "
          f"TRIALS={N_TRIALS} | WF_SPLITS={WF_SPLITS} | "
          f"LR=[{LR_LOW:.0e},{LR_HIGH:.0e}] | D=[{D_LOW},{D_HIGH}] | "
          f"NW=[{NW_LOW},{NW_HIGH}] | Stability-scoring | per-fold-scaler | financial-pruning ✓",
          flush=True)

    btc_raw = load_sync_resample(BTC_DIR, "BTC")
    eth_raw = load_sync_resample(ETH_DIR, "ETH")
    btc_raw, eth_raw = sync_assets(btc_raw, eth_raw); gc.collect()

    c_r  = btc_raw["close"]; h_r = btc_raw["high"]; l_r = btc_raw["low"]
    tr_r = pd.concat([(h_r-l_r),(h_r-c_r.shift()).abs(),(l_r-c_r.shift()).abs()],
                     axis=1).max(axis=1)
    atr_norm_u = (tr_r.ewm(com=13,adjust=False).mean()/(c_r+EPS)).astype(np.float32)

    feat_df, adx_aligned = mkfeat(btc_raw, eth_raw)
    close_s = btc_raw["close"].loc[feat_df.index]
    atr_s   = atr_norm_u.reindex(feat_df.index).ffill().bfill()
    del btc_raw, eth_raw, tr_r, c_r, h_r, l_r, atr_norm_u; gc.collect()

    n_seqs = len(feat_df) - SEQ_LEN - TB_BARS
    tr_cut, val_cut = _temporal_split(n_seqs)

    feat_arr = feat_df.values.astype(np.float32)
    del feat_df; gc.collect()

    _apply_log_transform(feat_arr)
    feat_arr_log = feat_arr.copy()

    train_end_raw = tr_cut + SEQ_LEN
    sc = fit_scaler(feat_arr, train_end_raw)
    gc.collect()

    c_arr   = close_s.values.astype(np.float64)
    atr_arr = atr_s.values.astype(np.float32)
    del close_s, atr_s; gc.collect()

    X, y, y_raw, atr_seqs = build_seqs(feat_arr, c_arr, atr_arr, adx_aligned)
    del feat_arr; gc.collect()

    Xtr = X[:tr_cut];          ytr = y[:tr_cut]
    Xv  = X[tr_cut:val_cut];   yv  = y[tr_cut:val_cut]
    Xte = X[val_cut:];         yte = y[val_cut:]
    yv_raw  = y_raw[tr_cut:val_cut];  yte_raw = y_raw[val_cut:]
    atr_v   = atr_seqs[tr_cut:val_cut]; atr_te = atr_seqs[val_cut:]

    dist = np.bincount(ytr, minlength=3)
    print(f"[labels] Short={dist[0]:,} | Neutral={dist[1]:,} | Long={dist[2]:,}", flush=True)
    print(f"[seqs]   train={len(Xtr):,} | val={len(Xv):,} | test={len(Xte):,}", flush=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda t: objective(t, Xtr, ytr, Xv, yv, yv_raw, atr_v, sc),
        n_trials=N_TRIALS, catch=(Exception,),
    )
    bp = study.best_params
    print(f"\n[optuna] best_params={bp}", flush=True)
    print_summary_table()
    tf.keras.backend.clear_session(); gc.collect()

    wfv_params = dict(units=bp["units"], nh=bp["nh"], d_rate=bp["d"],
                      lr=bp["lr"], noise=bp["noise"],
                      gru_layers=bp.get("gru_layers", 2))
    cw_best = compute_cw(ytr, neutral_weight=bp.get("neutral_weight", 1.5))

    wfv_metrics = walk_forward_validate(
        feat_arr_log, c_arr, atr_arr, adx_aligned, val_cut, wfv_params, cw_best
    )
    del feat_arr_log; gc.collect()

    with strategy.scope():
        m = build_model(SEQ_LEN, NF, **wfv_params)

    ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True)
    ds_vl = make_ds(Xv,  yv,  BATCH)

    cbs = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(os.path.join(OUT_DIR, "model_v13_best_final.keras"),
                        save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-7, verbose=1),
        TerminateOnNaN(),
    ]
    final_hist = m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=1,
                       callbacks=cbs, class_weight=cw_best)

    final_val_loss = float(min(final_hist.history.get("val_loss", [1e9])))

    pr    = m.predict(make_ds_eval(Xte, BATCH), verbose=0)
    n_min = min(len(pr), len(yte_raw), len(atr_te))
    stats = backtest(pr[:n_min], yte_raw[:n_min], atr_te[:n_min])
    FILE_METRICS.append(stats)

    wfv_fold_scores = wfv_metrics.get("_fold_scores", []) if wfv_metrics else []
    final_score     = _composite_score_v13(stats, wfv_scores=wfv_fold_scores,
                                            val_loss=final_val_loss)

    print(
        f"\n[FINAL TEST] Ret={stats['ret']:.2%} | WR={stats['wr']:.2%} | "
        f"Sharpe={stats['sharpe']:.2f} | Sortino={stats['sortino']:.2f} | "
        f"MDD={stats['mdd']:.2%} | PF={stats['pf']:.2f} | "
        f"Trades={stats['trades']} | TopRatio={stats['top_ratio']:.3f} | "
        f"ValLoss={final_val_loss:.4f} | FinalScore={final_score:.4f}",
        flush=True,
    )
    if wfv_metrics:
        print(f"[WFV  MEAN]  Ret={wfv_metrics.get('ret',0):.2%} | "
              f"WR={wfv_metrics.get('wr',0):.2%} | "
              f"Sortino={wfv_metrics.get('sortino',0):.2f} | "
              f"Sharpe={wfv_metrics.get('sharpe',0):.2f} | "
              f"MDD={wfv_metrics.get('mdd',0):.2%}", flush=True)

    print_stability_report(wfv_metrics)

    m.save(os.path.join(OUT_DIR, "model_v13_final.keras"))
    _try_save_best(m, final_score, sc, label="final")

    joblib.dump(
        {"sc": sc, "NF": NF, "N_PRICE": N_PRICE,
         "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS,
         "SEQ_LEN": SEQ_LEN, "SIGNAL_BUFFER": SIGNAL_BUFFER,
         "best_params": bp},
        os.path.join(OUT_DIR, "scaler_v13.pkl"),
    )
    joblib.dump(stats,         os.path.join(OUT_DIR, "stats_v13.save"))
    joblib.dump(wfv_metrics,   os.path.join(OUT_DIR, "wfv_metrics_v13.save"))
    joblib.dump(bp,            os.path.join(OUT_DIR, "best_params_v13.save"))
    joblib.dump(TRIAL_RESULTS, os.path.join(OUT_DIR, "trial_results_v13.save"))
    print(f"[done] → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    train()
