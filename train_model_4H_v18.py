import gc, os, glob, warnings, logging, random, json
import numpy as np, pandas as pd, tensorflow as tf, optuna, joblib
from numba import njit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import spearmanr
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GRU, Bidirectional, MultiHeadAttention, Add, GaussianNoise
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(False)
try: [tf.config.experimental.set_memory_growth(g, True) for g in tf.config.list_physical_devices("GPU")]
except Exception: pass
strategy = tf.distribute.MirroredStrategy()
NGPU = max(strategy.num_replicas_in_sync, 1)
BTC_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/BTC"
ETH_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/ETH"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)
SEQ_LEN = 60
BATCH = 128
EPOCHS = 60
N_TRIALS = 150
EPS = 1e-8
TP_MULT = 2.0
SL_MULT = 1.0
TB_BARS = 10
EMBARGO_BARS = SEQ_LEN + TB_BARS
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
WF_SPLITS = 5
WF_TRAIN_RATIO = 3
MAX_MDD = -0.25
MDD_SOFT = -0.18
MDD_LIN_THRESH = -0.10
TARGET_WR = 0.52
TARGET_SORTINO = 1.80
MIN_EDGE_MULTIPLE = 3.0
DYN_FEE_FLOOR = 0.0004
DYN_FEE_CAP = 0.0025
FEE_ATR_SCALE = 0.05
TOP_K_RATIO = 5
MIN_FEAT_CORR = 0.005
LOG_RET_PERIODS = [3, 6, 12, 24]
N_CLASSES = 3
_L2_DENSE = tf.keras.regularizers.l2(1e-5)
_L2_GRU = tf.keras.regularizers.l2(1e-3)
TRIAL_RESULTS = {}
BEST_SCORE = -np.inf
FILE_METRICS = []
LR_LOW, LR_HIGH = 1e-4, 3e-4
D_LOW, D_HIGH = 0.20, 0.40
NW_LOW, NW_HIGH = 1.0, 2.5
LONG_CONF_THRESHOLD = 0.58
SHORT_CONF_THRESHOLD = 0.52
LONG_PENALTY_DEFAULT = 0.72
PARQUET_CACHE_BTC = os.path.join(OUT_DIR, "ohlcv_btc_v18.parquet")
PARQUET_CACHE_ETH = os.path.join(OUT_DIR, "ohlcv_eth_v18.parquet")
PARQUET_CACHE_FEAT = os.path.join(OUT_DIR, "feat_v18.parquet")
PRICE_FEATS = [
    "close_norm","ema20_norm","ema50_norm","ema200_norm","rsi","rsi_slope","macd_norm",
    "macd_hist_norm","atr_norm","atr_regime","atr_breakout","price_range_norm","body_norm",
    "up_wick_norm","lo_wick_norm","bb_up_norm","bb_lo_norm","bb_w","tkr_ratio","rvi",
    "vol_trend","vpt_norm","log_ret","volatility","vol_zsc","roc","vol_scaled_ret",
    "eth_rsi_ratio","eth_close_norm","btc_eth_corr","obv_norm","vwap_ratio",
    "vwap_daily_dist","stoch_k","stoch_d","cci","cmf","adx","di_diff","adx_regime",
    "ichimoku_cloud","tenkan_kijun","vol_regime","close_above_ema200","delta_flow",
    "vol_imb_5","vol_imb_14","vol_imb_42","liq_proxy","mom_3","mom_6","mom_12","mom_24",
    "vol_quantile","atr_pct_ret","log_ret_3","log_ret_6","log_ret_12","log_ret_24",
    "daily_ret","daily_trend_strength","daily_ema_slope","price_vs_daily_open","is_gap",
    "ema200_slope","bear_market_depth","bear_chop_index",
    "bull_div_6","wpr14_reclaim","capitulation_red_vol","bb_reclaim",
]
_REGIME_FEAT_NAMES = [
    "adx","atr_norm","vol_regime","atr_regime","adx_regime","vol_quantile",
    "ema200_slope","bear_market_depth",
]
REGIME_IDX = [PRICE_FEATS.index(f) for f in _REGIME_FEAT_NAMES if f in PRICE_FEATS]
N_REGIME = len(REGIME_IDX)
TIME_FEATS = ["hour_sin","hour_cos","day_sin","day_cos","week_sin","week_cos","session_asia","session_eu","session_us","is_trending_session"]
ALL_FEATS = PRICE_FEATS + TIME_FEATS
NF = len(ALL_FEATS)
N_PRICE = len(PRICE_FEATS)
_LOG_SCALE_FEATS = ["tkr_ratio","vol_trend","vol_imb_5","vol_imb_14","vol_imb_42","atr_regime","price_range_norm"]
_LOG_SCALE_IDX = [PRICE_FEATS.index(f) for f in _LOG_SCALE_FEATS if f in PRICE_FEATS]
def _manifest(paths):
    result = []
    for p in paths:
        try:
            st = os.stat(p)
            result.append((os.path.basename(p), st.st_size, st.st_mtime))
        except OSError:
            result.append((os.path.basename(p), -1, -1.0))
    return result
def _load_resample_one(fp):
    try:
        df = pd.read_csv(fp, header=None, skiprows=1)
        nc = df.shape[1]
        if nc >= 8: df = df.iloc[:, [1,3,4,5,6,7]]
        elif nc == 7: df = df.iloc[:, [1,2,3,4,5,6]]
        elif nc == 6: df = df.iloc[:, :6]
        else: return None
        df.columns = ["timestamp","open","high","low","close","volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        df = df.sort_index()
        for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col], errors="coerce")
        r = df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        r["volume"] = r["volume"].fillna(0)
        return r.dropna(subset=["open","high","low","close"])
    except Exception: return None
def load_sync_resample(data_dir, label="", parquet_path=None):
    fps = sorted(set(glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)))
    if parquet_path and os.path.exists(parquet_path):
        meta_path = parquet_path + ".meta.json"
        current_manifest = _manifest(fps)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    saved_manifest = json.load(f)
                if saved_manifest == current_manifest:
                    df = pd.read_parquet(parquet_path)
                    print(f"[load] {label}: parquet cache hit ({len(df):,} rows)", flush=True)
                    return df
                else:
                    print(f"[load] {label}: manifest changed, rebuilding from CSVs...", flush=True)
            except Exception:
                print(f"[load] {label}: manifest read error, rebuilding from CSVs...", flush=True)
        else:
            print(f"[load] {label}: no manifest found, rebuilding from CSVs...", flush=True)
    parts = [ch for fp in fps for ch in [_load_resample_one(fp)] if ch is not None and len(ch) > 10]
    if not parts: raise RuntimeError(f"[load] {label}: no valid CSV in {data_dir}")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="4h"))
    df["is_gap"] = df["open"].isna().astype(np.float32)
    df["volume"] = df["volume"].fillna(0)
    df = df.ffill().dropna()
    df["vol"] = df["volume"]
    if parquet_path:
        df.to_parquet(parquet_path, compression="snappy")
        meta_path = parquet_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(_manifest(fps), f)
        print(f"[load] {label}: saved → {parquet_path}", flush=True)
    del parts; gc.collect()
    return df
def sync_assets(b, e):
    idx = b.index.intersection(e.index)
    return b.loc[idx].copy(), e.loc[idx].copy()
def _rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100 / (1 + g.ewm(com=p-1, adjust=False).mean() / (dn.ewm(com=p-1, adjust=False).mean() + EPS))
def _adx(h, l, c, p=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(com=p-1, adjust=False).mean()
    um = h-h.shift(); dm = l.shift()-l
    dmp = um.where((um>dm)&(um>0), 0.0); dmn = dm.where((dm>um)&(dm>0), 0.0)
    dip = 100*dmp.ewm(com=p-1, adjust=False).mean()/(a+EPS); din = 100*dmn.ewm(com=p-1, adjust=False).mean()/(a+EPS)
    dx = 100*(dip-din).abs()/(dip+din+EPS)
    return dx.ewm(com=p-1, adjust=False).mean(), dip-din
def mkfeat(btc_df, eth_df, parquet_path=None, btc_parquet_path=None, eth_parquet_path=None):
    if parquet_path and os.path.exists(parquet_path):
        meta_path = parquet_path + ".meta.json"
        upstream_paths = [p for p in [btc_parquet_path, eth_parquet_path] if p and os.path.exists(p)]
        current_manifest = _manifest(upstream_paths)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    saved_manifest = json.load(f)
                if saved_manifest == current_manifest:
                    cached = pd.read_parquet(parquet_path)
                    if list(cached.columns) == ALL_FEATS:
                        print(f"[mkfeat] parquet cache hit ({len(cached):,} rows)", flush=True)
                        return cached[ALL_FEATS].astype(np.float32)
                    print(f"[mkfeat] cache schema mismatch, rebuilding...", flush=True)
                    os.remove(parquet_path)
                else:
                    print(f"[mkfeat] manifest changed, rebuilding...", flush=True)
                    os.remove(parquet_path)
            except Exception:
                print(f"[mkfeat] manifest read error, rebuilding...", flush=True)
                if os.path.exists(parquet_path): os.remove(parquet_path)
        else:
            print(f"[mkfeat] no manifest found, rebuilding...", flush=True)
            os.remove(parquet_path)
    df = btc_df.astype(np.float64).copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]
    ec = eth_df["close"].reindex(df.index, method="ffill").ffill().astype(np.float64)
    ema20 = c.ewm(span=20, adjust=False).mean(); ema50 = c.ewm(span=50, adjust=False).mean(); ema200 = c.ewm(span=200, adjust=False).mean()
    df["close_norm"] = c / (c.rolling(200).mean().replace(0, np.nan).ffill() + EPS)
    df["ema20_norm"] = ema20 / (c + EPS) - 1.0; df["ema50_norm"] = ema50 / (c + EPS) - 1.0; df["ema200_norm"] = ema200 / (c + EPS) - 1.0
    rsi_raw = _rsi(c); df["rsi"] = rsi_raw; df["rsi_slope"] = rsi_raw.diff(5) / 5.0
    df["eth_rsi_ratio"] = (_rsi(ec) / (rsi_raw + EPS)).clip(-10.0, 10.0)
    df["eth_close_norm"] = ec / (ec.rolling(200).mean().replace(0, np.nan).ffill() + EPS)
    c_safe = c.clip(lower=EPS)
    ec_safe = ec.clip(lower=EPS)
    br = np.log(c_safe) - np.log(c_safe.shift(1).clip(lower=EPS))
    er = np.log(ec_safe) - np.log(ec_safe.shift(1).clip(lower=EPS))
    df["log_ret"] = br; df["btc_eth_corr"] = br.rolling(42).corr(er).fillna(0).clip(-1, 1)
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd_r = e12 - e26; sig_r = macd_r.ewm(span=9, adjust=False).mean()
    df["macd_norm"] = macd_r / (c + EPS); df["macd_hist_norm"] = (macd_r - sig_r) / (c + EPS)
    tr = pd.concat([(h - l),(h - c.shift()).abs(),(l - c.shift()).abs()], axis=1).max(axis=1)
    atr_raw = tr.ewm(com=13, adjust=False).mean()
    df["atr_norm"] = atr_raw / (c + EPS); df["atr_regime"] = atr_raw / (atr_raw.rolling(42).mean() + EPS)
    df["atr_breakout"] = (c - c.rolling(20).mean()) / (atr_raw + EPS); df["price_range_norm"] = (h - l) / (atr_raw + EPS)
    df["atr_pct_ret"] = br / (df["atr_norm"] + EPS); df["body_norm"] = (c - o).abs() / (atr_raw + EPS)
    oc_hi = pd.concat([o, c], axis=1).max(axis=1); oc_lo = pd.concat([o, c], axis=1).min(axis=1)
    df["up_wick_norm"] = (h - oc_hi) / (atr_raw + EPS); df["lo_wick_norm"] = (oc_lo - l) / (atr_raw + EPS)
    s20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    df["bb_up_norm"] = (s20 + 2*sd20 - c) / (c + EPS); df["bb_lo_norm"] = (s20 - 2*sd20 - c) / (c + EPS); df["bb_w"] = (4*sd20) / (s20 + EPS)
    ve20 = v.ewm(span=20, adjust=False).mean()
    df["tkr_ratio"] = v / (ve20 + EPS); df["rvi"] = (v - v.rolling(42).mean()) / (v.rolling(42).std() + EPS)
    df["vol_trend"] = v.ewm(span=12, adjust=False).mean() / (v.ewm(span=42, adjust=False).mean() + EPS)
    vpt = (br * v).cumsum()
    df["vpt_norm"] = (vpt / (vpt.rolling(200).std() + EPS)).replace([np.inf, -np.inf], 0.0)
    df["volatility"] = br.rolling(14).std(); df["roc"] = (c - c.shift(14)) / (c.shift(14) + EPS)
    df["vol_scaled_ret"] = (br / (df["volatility"] + EPS)).replace([np.inf, -np.inf], 0.0)
    eth_vol = er.rolling(14).std(); pool = (df["volatility"] + eth_vol) / 2
    df["vol_zsc"] = (df["volatility"] - pool.rolling(180).mean()) / (pool.rolling(180).std() + EPS)
    obv = pd.Series(np.cumsum(np.sign(br.fillna(0).values) * v.values), index=df.index)
    df["obv_norm"] = (obv / (obv.rolling(200).std() + EPS)).replace([np.inf, -np.inf], 0.0)
    tp_v = (h + l + c) / 3; df["vwap_ratio"] = c / ((tp_v * v).rolling(20).sum() / (v.rolling(20).sum() + EPS) + EPS)
    vd = (tp_v * v).groupby(pd.Grouper(freq="D")).cumsum() / (v.groupby(pd.Grouper(freq="D")).cumsum() + EPS)
    df["vwap_daily_dist"] = (c - vd) / (c + EPS)
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["stoch_k"] = (c - lo14) / (hi14 - lo14 + EPS) * 100; df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["cci"] = (tp_v - tp_v.rolling(20).mean()) / (0.015 * tp_v.rolling(20).std() + EPS)
    mfv = ((2*c - h - l) / (h - l + EPS)) * v; df["cmf"] = mfv.rolling(20).sum() / (v.rolling(20).sum() + EPS)
    adx_v, di_d = _adx(h, l, c, p=14)
    df["adx"] = adx_v; df["di_diff"] = di_d; df["adx_regime"] = (adx_v > 18.0).astype(np.float64)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2; kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    s_a = (tenkan + kijun) / 2; s_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    df["ichimoku_cloud"] = (s_a - s_b) / (c + EPS); df["tenkan_kijun"] = (tenkan - kijun) / (c + EPS)
    v30 = df["volatility"].rolling(180).mean(); v30s = df["volatility"].rolling(180).std()
    df["vol_regime"] = (df["volatility"] - v30) / (v30s + EPS); df["close_above_ema200"] = (c > ema200).astype(np.float64)
    up_v = v.where(br > 0, 0.0); dn_v = v.where(br < 0, 0.0)
    df["delta_flow"] = (up_v - dn_v).rolling(20).mean() / (v.rolling(20).mean() + EPS)
    df["vol_imb_5"] = up_v.rolling(5).sum() / (dn_v.rolling(5).sum() + EPS)
    df["vol_imb_14"] = up_v.rolling(14).sum() / (dn_v.rolling(14).sum() + EPS)
    df["vol_imb_42"] = up_v.rolling(42).sum() / (dn_v.rolling(42).sum() + EPS)
    wr_f = (df["up_wick_norm"] + df["lo_wick_norm"]) / (df["body_norm"] + EPS)
    df["liq_proxy"] = ((df["atr_regime"] > 1.5).astype(float) * (df["tkr_ratio"] > 2.0).astype(float) * (wr_f > 1.5).astype(float)).astype(np.float64)
    df["mom_3"] = (c - c.shift(3)) / (c.shift(3) + EPS); df["mom_6"] = (c - c.shift(6)) / (c.shift(6) + EPS)
    df["mom_12"] = (c - c.shift(12)) / (c.shift(12) + EPS); df["mom_24"] = (c - c.shift(24)) / (c.shift(24) + EPS)
    df["vol_quantile"] = df["volatility"].rolling(360).rank(pct=True)
    for p in LOG_RET_PERIODS: df[f"log_ret_{p}"] = np.log(c_safe) - np.log(c_safe.shift(p).clip(lower=EPS))
    c_24h_ago = c.shift(6); atr_24h_mean = atr_raw.rolling(6).mean(); ema_daily_proxy = c.ewm(span=120, adjust=False).mean()
    df["daily_ret"] = np.log((c + EPS) / (c_24h_ago + EPS))
    df["daily_trend_strength"] = ((c - c_24h_ago) / (atr_24h_mean + EPS)).clip(-5, 5)
    df["daily_ema_slope"] = (ema_daily_proxy - ema_daily_proxy.shift(6)) / (ema_daily_proxy.shift(6) + EPS)
    df["price_vs_daily_open"] = (c - c_24h_ago) / (c + EPS)
    df["is_gap"] = btc_df["is_gap"].reindex(df.index).fillna(0).astype(np.float64) if "is_gap" in btc_df.columns else 0.0
    ema200_slope_raw = (ema200 - ema200.shift(50)) / (ema200.shift(50).abs() + EPS)
    df["ema200_slope"] = ema200_slope_raw.clip(-0.05, 0.05).astype(np.float64)
    df["bear_market_depth"] = ((ema200 - c) / (atr_raw + EPS)).clip(-10.0, 10.0).astype(np.float64)
    bear_flag = (c < ema200).astype(float)
    chop_flag = (adx_v < 18.0).astype(float)
    df["bear_chop_index"] = (bear_flag * chop_flag).astype(np.float64)
    del c_24h_ago, atr_24h_mean, ema_daily_proxy, up_v, dn_v, obv, vpt, bear_flag, chop_flag
    p6 = np.log(c_safe / c_safe.shift(6).clip(lower=EPS))
    r6 = (rsi_raw - rsi_raw.shift(6)) / 100.0
    df["bull_div_6"] = np.tanh((-p6) * r6 * 8.0).astype(np.float32)
    hi14 = h.rolling(14).max()
    lo14 = l.rolling(14).min()
    wpr14 = -100.0 * (hi14 - c) / (hi14 - lo14 + EPS)
    df["wpr14_reclaim"] = (np.tanh((wpr14 + 80.0) / 20.0) * np.tanh((rsi_raw - 30.0) / 10.0)).astype(np.float32)
    vol20 = v.rolling(20).mean()
    body = (c - o).abs() / (atr_raw + EPS)
    red = (c < o).astype(np.float32)
    vol_spike = (v / (vol20 + EPS) - 1.0).clip(0.0, 5.0)
    lower_wick = ((np.minimum(o, c) - l) / (h - l + EPS)).clip(0.0, 1.0)
    df["capitulation_red_vol"] = (red * vol_spike * lower_wick / (body + EPS)).astype(np.float32)
    bb_low = s20 - 2.0 * sd20
    prev_below = (c.shift(1) < bb_low.shift(1)).astype(np.float32)
    reclaim = (c > bb_low).astype(np.float32)
    df["bb_reclaim"] = (prev_below * reclaim * np.tanh((c - bb_low) / (atr_raw + EPS))).astype(np.float32)
    hr = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24); df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7); df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    how = df.index.dayofweek * 24 + hr
    df["week_sin"] = np.sin(2 * np.pi * how / 168); df["week_cos"] = np.cos(2 * np.pi * how / 168)
    df["session_asia"] = ((hr >= 0) & (hr < 8)).astype(np.float64)
    df["session_eu"] = ((hr >= 8) & (hr < 16)).astype(np.float64)
    df["session_us"] = ((hr >= 16) & (hr < 24)).astype(np.float64)
    df["is_trending_session"] = ((hr >= 8) & (hr < 20)).astype(np.float64)
    df[PRICE_FEATS] = df[PRICE_FEATS].shift(1)
    df = df.dropna(subset=ALL_FEATS)[ALL_FEATS].astype(np.float32)
    if parquet_path:
        df.to_parquet(parquet_path, compression="snappy")
        meta_path = parquet_path + ".meta.json"
        upstream_paths = [p for p in [btc_parquet_path, eth_parquet_path] if p and os.path.exists(p)]
        with open(meta_path, "w") as f:
            json.dump(_manifest(upstream_paths), f)
        print(f"[mkfeat] saved → {parquet_path} ({df.memory_usage(deep=True).sum()/1e6:.1f} MB)", flush=True)
    return df
def _apply_log_transform(feat_arr):
    np.nan_to_num(feat_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    for idx in _LOG_SCALE_IDX:
        col = feat_arr[:, idx]
        feat_arr[:, idx] = np.sign(col) * np.log1p(np.abs(col)).astype(np.float32)
def fit_scale_inplace(feat_arr, train_end_raw):
    sc = RobustScaler()
    feat_arr[:train_end_raw, :N_PRICE] = sc.fit_transform(feat_arr[:train_end_raw, :N_PRICE])
    feat_arr[train_end_raw:, :N_PRICE] = sc.transform(feat_arr[train_end_raw:, :N_PRICE])
    return sc
def apply_scale_transform(feat_arr, sc):
    feat_arr[:, :N_PRICE] = sc.transform(feat_arr[:, :N_PRICE])
@njit(cache=True)
def _compute_barrier_returns_nb(open_arr, high_arr, low_arr, close_arr, atr_arr, sl, n_seqs, tp_m, sl_m, nb, eps, fee_floor, fee_cap, fee_atr_scale):
    barrier_rets = np.zeros(n_seqs, dtype=np.float64)
    intrabar_flag = np.zeros(n_seqs, dtype=np.bool_)
    labels = np.ones(n_seqs, dtype=np.int32)
    hold_bars = np.zeros(n_seqs, dtype=np.int32)
    for i in range(n_seqs):
        bar = sl + i - 1
        entry = open_arr[bar + 1]
        if entry <= 0.0: continue
        atr_i = atr_arr[bar]
        dyn_fee = min(fee_floor + atr_i * fee_atr_scale, fee_cap)
        tp_lvl = entry * (1.0 + tp_m * atr_i); sl_lvl = entry * (1.0 - sl_m * atr_i)
        tp_hit = nb; sl_hit = nb
        for j in range(1, nb + 1):
            h = high_arr[bar + j]; l = low_arr[bar + j]
            if tp_hit == nb and h >= tp_lvl: tp_hit = j - 1
            if sl_hit == nb and l <= sl_lvl: sl_hit = j - 1
            if tp_hit < nb and sl_hit < nb: break
        if tp_hit == nb and sl_hit == nb:
            exit_price = close_arr[bar + nb]; hold_bars[i] = nb
        elif tp_hit < sl_hit:
            exit_price = tp_lvl; labels[i] = 2; hold_bars[i] = tp_hit + 1
        elif sl_hit < tp_hit:
            exit_price = sl_lvl; labels[i] = 0; hold_bars[i] = sl_hit + 1
        else:
            exit_price = sl_lvl; intrabar_flag[i] = True; labels[i] = 0; hold_bars[i] = sl_hit + 1
        barrier_rets[i] = np.log(exit_price / (entry + eps)) - 2.0 * dyn_fee
    return labels, barrier_rets, intrabar_flag, hold_bars
def compute_sample_weights(y_raw, intrabar_mask):
    w = np.log1p(np.abs(y_raw) * 100.0).astype(np.float32)
    if intrabar_mask is not None: w[intrabar_mask] = 0.0
    return w
def _apply_gap_mask(sw, gap_arr, sl, n_seqs):
    if gap_arr is None or len(gap_arr) == 0: return sw
    gap_slice = gap_arr[sl - 1 : sl - 1 + n_seqs + TB_BARS - 1]
    if len(gap_slice) < n_seqs + TB_BARS - 1: return sw
    windows = np.lib.stride_tricks.sliding_window_view(gap_slice[:n_seqs + TB_BARS - 1], TB_BARS)
    mask = windows.max(axis=1) > 0
    sw = sw.copy(); sw[mask[:n_seqs]] = 0.0
    return sw
def _build_labels(open_arr, high_arr, low_arr, close_arr, atr_arr, sl=SEQ_LEN):
    n_seqs = len(open_arr) - sl - TB_BARS + 1
    labels, y_raw, intrabar, hold_bars = _compute_barrier_returns_nb(
        open_arr.astype(np.float64), high_arr.astype(np.float64),
        low_arr.astype(np.float64), close_arr.astype(np.float64),
        atr_arr.astype(np.float64), sl, n_seqs,
        float(TP_MULT), float(SL_MULT), int(TB_BARS),
        EPS, DYN_FEE_FLOOR, DYN_FEE_CAP, FEE_ATR_SCALE)
    return labels, y_raw, intrabar, hold_bars, n_seqs
def _build_X_seqs(feat_arr, n_seqs, disabled_feats=None, sl=SEQ_LEN):
    idx = np.arange(n_seqs)[:, None] + np.arange(sl)
    X = feat_arr[idx]
    if disabled_feats: X = X.copy(); X[:, :, disabled_feats] = 0.0
    return X
def build_seqs(feat_arr, open_arr, high_arr, low_arr, close_arr, atr_arr, gap_arr=None, disabled_feats=None, sl=SEQ_LEN):
    labels, y_raw, intrabar, hold_bars, n_seqs = _build_labels(open_arr, high_arr, low_arr, close_arr, atr_arr, sl)
    X = _build_X_seqs(feat_arr, n_seqs, disabled_feats, sl)
    y = labels.astype(np.int32)
    atr_s = atr_arr[sl - 1 : sl - 1 + n_seqs]
    sw = compute_sample_weights(y_raw, intrabar)
    sw = _apply_gap_mask(sw, gap_arr, sl, n_seqs)
    return X, y, y_raw.astype(np.float32), atr_s, sw, hold_bars
def select_features(X_train, y_raw_train, threshold=MIN_FEAT_CORR):
    n_feat = X_train.shape[2]; n_sub = min(len(X_train), 5000)
    rng = np.random.default_rng(SEED); idx = rng.choice(len(X_train), n_sub, replace=False)
    last = X_train[idx, -1, :]; mid = X_train[idx, X_train.shape[1] // 2, :]
    y_sub = y_raw_train[idx]; disabled = []
    for fi in range(n_feat):
        try:
            col_last = last[:, fi]; col_mid = mid[:, fi]
            if col_last.std() < 1e-7: disabled.append(fi); continue
            r_last, _ = spearmanr(col_last, y_sub)
            r_mid, _ = spearmanr(col_mid, y_sub)
            if np.isnan(r_last) and np.isnan(r_mid): disabled.append(fi); continue
            best_corr = max(0.0 if np.isnan(r_last) else abs(r_last), 0.0 if np.isnan(r_mid) else abs(r_mid))
            if best_corr < threshold: disabled.append(fi)
        except Exception: disabled.append(fi)
    return disabled
def _temporal_split(n): return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
def compute_cw(ytr, neutral_weight=1.5, long_penalty=0.75):
    classes = np.array([0, 1, 2])
    counts = np.bincount(ytr, minlength=3).astype(np.float32)
    if np.any(counts == 0):
        raw_w = np.ones(3, dtype=np.float32)
        present = counts > 0
        raw_w[present] = counts[present].sum() / (present.sum() * counts[present])
    else:
        raw_w = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    raw_w[1] *= float(neutral_weight)
    raw_w[2] *= float(long_penalty)
    raw_w /= raw_w.mean()
    print(f"[cw nw={neutral_weight:.2f} lp={long_penalty:.2f}] Short={raw_w[0]:.3f} | Neutral={raw_w[1]:.3f} | Long={raw_w[2]:.3f}", flush=True)
    return {i: float(raw_w[i]) for i in range(3)}
def apply_class_weights(sw, y, cw_dict):
    cw_arr = np.array([cw_dict[int(yi)] for yi in y], dtype=np.float32)
    return (sw * cw_arr).astype(np.float32)
def apply_asymmetric_thresholds(pr, long_thr=LONG_CONF_THRESHOLD, short_thr=SHORT_CONF_THRESHOLD):
    pr_adj = pr.copy().astype(np.float32)
    long_below = pr_adj[:, 2] < long_thr
    pr_adj[long_below, 2] = 0.0
    short_below = pr_adj[:, 0] < short_thr
    pr_adj[short_below, 0] = 0.0
    return pr_adj
class TemporalAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, recent_bias=2.0, **kwargs):
        super().__init__(**kwargs); self.recent_bias = recent_bias
        self._score = Dense(1, use_bias=True, dtype="float32")
    def call(self, x, training=None):
        x_f32 = tf.cast(x, tf.float32)
        seq_len = tf.shape(x_f32)[1]
        scores = self._score(x_f32)
        pos = tf.cast(tf.range(seq_len), tf.float32) / tf.cast(tf.maximum(seq_len - 1, 1), tf.float32)
        pos = tf.reshape(pos, (1, -1, 1))
        scores = scores + tf.cast(self.recent_bias, tf.float32) * pos
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(x_f32 * weights, axis=1)
        return tf.cast(context, x.dtype)
    def get_config(self):
        cfg = super().get_config(); cfg.update({"recent_bias": self.recent_bias}); return cfg
class FiLMLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs); self.units = units
        self._h = Dense(units // 2, activation="elu")
        self._gamma = Dense(units, use_bias=False, kernel_initializer="zeros")
        self._beta = Dense(units, use_bias=True, bias_initializer="zeros")
    def call(self, x, regime, training=None):
        h = self._h(regime); gamma = self._gamma(h) + 1.0; beta = self._beta(h)
        return gamma[:, tf.newaxis, :] * x + beta[:, tf.newaxis, :]
    def get_config(self):
        cfg = super().get_config(); cfg["units"] = self.units; return cfg
def build_model(seq_len, n_feat, units, nh, d_rate, lr, noise, gru_layers=2):
    inp = Input(shape=(seq_len, n_feat), dtype="float32")
    regime = tf.keras.layers.Lambda(lambda x: tf.gather(x[:, -1, :], REGIME_IDX, axis=-1))(inp)
    x = GaussianNoise(noise)(inp); x = Dense(units, kernel_regularizer=_L2_DENSE)(x)
    for _ in range(gru_layers):
        res = x; x = LayerNormalization()(x)
        x = Bidirectional(GRU(units // 2, return_sequences=True, dropout=d_rate, recurrent_dropout=0.0, kernel_regularizer=_L2_GRU, recurrent_regularizer=_L2_GRU))(x)
        x = Add()([x, res])
    res = x; x_n = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=nh, key_dim=max(units // nh, 8), dropout=d_rate)(x_n, x_n)
    attn = FiLMLayer(units)(attn, regime); x = Add()([attn, res])
    x = LayerNormalization()(x); x = TemporalAttentionPooling(recent_bias=2.0)(x)
    x = Dense(units // 2, activation="elu", kernel_regularizer=_L2_DENSE)(x); x = Dropout(d_rate)(x)
    out = Dense(N_CLASSES, activation="softmax", dtype="float32")(x)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
def make_ds(X, y, w, batch, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y, w))
    if shuffle: ds = ds.shuffle(min(len(y), 50_000), reshuffle_each_iteration=True)
    return ds.batch(batch * NGPU, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)
def make_ds_eval(X, batch):
    return tf.data.Dataset.from_tensor_slices(X).batch(batch * NGPU, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
@njit(cache=True)
def _run_stateful_backtest(sig, y_raw, hold_bars):
    n = len(sig); rets = np.zeros(n, dtype=np.float32); trade_flags = np.zeros(n, dtype=np.float32); next_free = 0
    for i in range(n):
        if i < next_free or sig[i] == 0.0: continue
        rets[i] = sig[i] * y_raw[i]; trade_flags[i] = 1.0; next_free = i + hold_bars[i] + 1
    return rets, trade_flags
def backtest(proba, y_raw, hold_bars):
    _default = dict(ret=0.0, mdd=0.0, sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0, trades=0, top_ratio=0.0, long_wr=0.0, short_wr=0.0, long_ret=0.0, short_ret=0.0, long_trades=0, short_trades=0)
    try:
        n = min(len(proba), len(y_raw), len(hold_bars))
        if n == 0: return _default
        proba = proba[:n]; y_raw = y_raw[:n]; hb_ = hold_bars[:n]
        pred = np.argmax(proba, axis=1); conf = np.max(proba, axis=1)
        raw_sig = np.where(pred == 2, 1.0, np.where(pred == 0, -1.0, 0.0))
        size = np.clip((conf - 0.52) / 0.48, 0.0, 1.0); sig = raw_sig * size
        rets, trade_flag = _run_stateful_backtest(sig.astype(np.float32), y_raw.astype(np.float32), hb_.astype(np.int32))
        if len(rets) == 0 or np.sum(trade_flag) == 0: return _default
        cum = np.cumprod(1.0 + np.clip(rets, -0.99, 10.0)); pk = np.maximum.accumulate(cum)
        mdd = float(((cum - pk) / (pk + EPS)).min())
        active_rets = rets[trade_flag > 0]; n_tr = int(np.sum(trade_flag))
        trades_per_year = n_tr / max(n / (365.0 * 6.0), EPS)
        ann = float(np.sqrt(max(trades_per_year, 1.0)))
        ret_std = float(active_rets.std()) if len(active_rets) > 1 else EPS
        sh = float(active_rets.mean() / (ret_std + EPS) * ann)
        neg = active_rets[active_rets < 0]; neg_std = float(neg.std()) if len(neg) > 1 else EPS
        sortino = float(active_rets.mean() / (neg_std + EPS) * ann)
        pos_sum = float(active_rets[active_rets > 0].sum()) if (active_rets > 0).any() else 0.0
        neg_sum = float(abs(neg.sum())) if len(neg) > 0 else EPS; pf = pos_sum / (neg_sum + EPS)
        wr = float(np.sum(active_rets > 0)) / max(n_tr, 1)
        pos_rets = active_rets[active_rets > 0]
        if len(pos_rets) >= TOP_K_RATIO: top_k = np.sort(pos_rets)[-TOP_K_RATIO:]; top_ratio = float(top_k.sum() / (pos_rets.sum() + EPS))
        elif len(pos_rets) > 0: top_ratio = float(pos_rets.max() / (pos_rets.sum() + EPS))
        else: top_ratio = 0.0
        long_mask = (pred == 2) & (trade_flag > 0); short_mask = (pred == 0) & (trade_flag > 0)
        long_rets_v = rets[long_mask]; short_rets_v = rets[short_mask]
        long_wr = float(np.mean(long_rets_v > 0)) if len(long_rets_v) > 0 else 0.0
        short_wr = float(np.mean(short_rets_v > 0)) if len(short_rets_v) > 0 else 0.0
        long_ret = float(long_rets_v.sum()) if len(long_rets_v) > 0 else 0.0
        short_ret = float(short_rets_v.sum()) if len(short_rets_v) > 0 else 0.0
        return dict(ret=float(cum[-1]-1), mdd=mdd, sharpe=sh, sortino=sortino, pf=pf, wr=wr, trades=n_tr, top_ratio=top_ratio, long_wr=long_wr, short_wr=short_wr, long_ret=long_ret, short_ret=short_ret, long_trades=int(np.count_nonzero(long_mask)), short_trades=int(np.count_nonzero(short_mask)))
    except Exception as e: print(f"[backtest] ERROR: {e}", flush=True); return _default
def _fold_score(bt):
    if bt["trades"] == 0: return -2.0
    sortino_s = float(np.clip(bt["sortino"], -3.0, 15.0)); pf_s = float(min(bt["pf"], 10.0)); ret_s = float(np.log(max(bt["ret"] + 1.0, 1e-6)))
    mdd_abs = float(abs(bt["mdd"])); lin_t = abs(MDD_LIN_THRESH); quad_t = abs(MDD_SOFT)
    if mdd_abs <= lin_t: mdd_pen = mdd_abs * 0.5
    elif mdd_abs <= quad_t: mdd_pen = lin_t * 0.5 + (mdd_abs - lin_t)**2 * 10.0
    else: mdd_pen = lin_t * 0.5 + (quad_t - lin_t)**2 * 10.0 + (mdd_abs - quad_t) * 25.0
    return float((sortino_s * 0.40) + (pf_s * 0.35) + (ret_s * 0.25) - mdd_pen)
def _composite_score(bt, wfv_scores=None, val_loss=0.0):
    if bt["trades"] == 0: return -2.0
    if wfv_scores and len(wfv_scores) > 0:
        arr = np.array(wfv_scores, dtype=np.float64)
        mean_s = float(np.mean(arr)); median_s = float(np.median(arr))
        min_s = float(np.min(arr)); std_s = float(np.std(arr))
        score = 0.55 * mean_s + 0.20 * median_s + 0.15 * min_s - 0.10 * std_s
        if min_s < -0.25: score -= 0.75
    else:
        ret = bt["ret"]
        mdd_abs = abs(bt["mdd"])
        n_tr = max(bt["trades"], 1)
        prof_rate = bt["wr"]
        ret_pos = np.log1p(max(ret, 0.0))
        ret_neg = np.log1p(max(-ret, 0.0))
        ret_term = 3.5 * ret_pos - 1.0 * ret_neg
        prof_term = 2.0 * (prof_rate - 0.50)
        pf_term = 0.4 * np.log1p(min(bt["pf"], 10.0))
        wr_term = 0.25 * (bt["wr"] - 0.52)
        if mdd_abs <= 0.10: mdd_pen = 0.0
        elif mdd_abs <= 0.20: mdd_pen = 0.6 * ((mdd_abs - 0.10) / 0.10)**2
        else: mdd_pen = 0.6 + 3.0 * (mdd_abs - 0.20)
        long_bonus = 0.0
        if bt.get("long_trades", 0) >= 10:
            long_wr = bt.get("long_wr", 0.0)
            long_bonus = 0.8 * np.tanh((long_wr - 0.42) / 0.08)
        score = ret_term + prof_term + pf_term + wr_term + long_bonus - mdd_pen
        long_share = bt.get("long_trades", 0) / n_tr
        if long_share < 0.25: score -= 1.5 * (0.25 - long_share)
        elif long_share > 0.60: score -= 0.25 * (long_share - 0.60)
    score -= 0.1 * float(val_loss)
    return float(score)
def _try_save_best(model, score, label=""):
    global BEST_SCORE
    if np.isfinite(score) and score > BEST_SCORE:
        BEST_SCORE = score; model.save(os.path.join(OUT_DIR, "best_model_v18.keras"))
        print(f"  ★ [{label}] NEW BEST  score={score:.4f} → best_model_v18.keras", flush=True)
def objective(trial, Xtr, ytr, sw_tr, Xv, yv, sw_v, yv_raw, hb_v):
    tf.keras.backend.clear_session(); tf.compat.v1.reset_default_graph(); gc.collect()
    units = trial.suggest_categorical("units", [128, 256, 384, 512])
    nh = trial.suggest_categorical("nh", [4, 8, 16])
    d = trial.suggest_float("d", D_LOW, D_HIGH)
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    noise = trial.suggest_float("noise", 0.002, 0.03)
    gru_layers = trial.suggest_categorical("gru_layers", [1, 2])
    neutral_weight = trial.suggest_float("neutral_weight", NW_LOW, NW_HIGH)
    long_penalty = trial.suggest_float("long_penalty", 0.95, 1.25)
    long_conf_thr = trial.suggest_float("long_conf_thr", 0.40, 0.58)
    short_conf_thr = trial.suggest_float("short_conf_thr", 0.52, 0.66)
    if units % nh != 0: return float("inf")
    if short_conf_thr < long_conf_thr + 0.04: return float("inf")
    tn = trial.number + 1; opt_score = float("inf")
    m = None; pr = None; hist = None; ds_tr = None; ds_vl = None; sw_tr_w = None; sw_v_w = None; cw_trial = None
    try:
        cw_trial = compute_cw(ytr, neutral_weight=neutral_weight, long_penalty=long_penalty)
        sw_tr_w = apply_class_weights(sw_tr, ytr, cw_trial)
        sw_v_w = apply_class_weights(sw_v, yv, cw_trial)
        with strategy.scope(): m = build_model(SEQ_LEN, NF, units, nh, d, lr, noise, gru_layers)
        ds_tr = make_ds(Xtr, ytr, sw_tr_w, BATCH, shuffle=True); ds_vl = make_ds(Xv, yv, sw_v_w, BATCH)
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        cb = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=0), TerminateOnNaN(), es]
        hist = m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=0, callbacks=cb)
        final_val_loss = min(hist.history["val_loss"])
        pr = m.predict(make_ds_eval(Xv, BATCH), verbose=0)
        n_min = min(len(pr), len(yv_raw), len(hb_v))
        pr_gated = apply_asymmetric_thresholds(pr[:n_min], long_thr=long_conf_thr, short_thr=short_conf_thr)
        bt = backtest(pr_gated, yv_raw[:n_min], hb_v[:n_min])
        score = _composite_score(bt, wfv_scores=None, val_loss=final_val_loss); opt_score = -score
        if np.isfinite(score):
            ckpt = os.path.join(OUT_DIR, f"model_v18_trial_{tn:02d}.keras"); m.save(ckpt)
            mdd_tag = "🔴" if bt["mdd"] < MAX_MDD else ("⚠️" if bt["mdd"] < MDD_SOFT else "✅")
            TRIAL_RESULTS[tn] = {"score": score, "ret": bt["ret"], "sharpe": bt["sharpe"], "sortino": bt["sortino"], "mdd": bt["mdd"], "pf": bt["pf"], "wr": bt["wr"], "trades": bt["trades"], "top_ratio": bt["top_ratio"], "long_wr": bt["long_wr"], "short_wr": bt["short_wr"], "neutral_weight": neutral_weight, "long_penalty": long_penalty, "long_conf_thr": long_conf_thr, "short_conf_thr": short_conf_thr, "model": ckpt, "val_loss": final_val_loss}
            print(f"[T{tn:02d}/{N_TRIALS}] u={units}|nh={nh}|gl={gru_layers}|d={d:.2f}|lr={lr:.1e}|nw={neutral_weight:.2f}|lp={long_penalty:.2f}|lthr={long_conf_thr:.2f} | Ret={bt['ret']:.2%}|WR={bt['wr']:.2%}|L={bt['long_wr']:.2%}|S={bt['short_wr']:.2%}|Sortino={bt['sortino']:.2f}|MDD={bt['mdd']:.2%}|TR5={bt['top_ratio']:.3f}|score={score:.4f} {mdd_tag}", flush=True)
            _try_save_best(m, score, label=f"T{tn:02d}")
    except Exception as e: print(f"[T{tn:02d}] ERROR: {e}", flush=True)
    finally:
        del m, pr, hist, ds_tr, ds_vl, sw_tr_w, sw_v_w, cw_trial
        tf.keras.backend.clear_session(); tf.compat.v1.reset_default_graph(); gc.collect()
    return opt_score
def walk_forward_validate(feat_arr_log, o_arr, h_arr, l_arr, c_arr, atr_arr, gap_arr, n_wfv, params, cw, disabled_feats):
    wf_val_sz = (n_wfv - EMBARGO_BARS * WF_SPLITS) // (WF_SPLITS + 1)
    wf_train_sz = wf_val_sz * WF_TRAIN_RATIO; wf_metrics = []; fold_scores = []
    _lthr = params.get("long_conf_thr", LONG_CONF_THRESHOLD)
    _sthr = params.get("short_conf_thr", SHORT_CONF_THRESHOLD)
    model_params = {k: v for k, v in params.items() if k not in ("long_conf_thr", "short_conf_thr")}
    print(f"\n[WFV-v18] {WF_SPLITS} folds | val_sz≈{wf_val_sz:,} | train_sz≈{wf_train_sz:,} | embargo={EMBARGO_BARS} bars | anchored-expanding→sliding ✓ | scaler-isolated ✓ | gap-masked ✓", flush=True)
    for fold in range(WF_SPLITS):
        train_end = wf_val_sz * (fold + 1); train_start = max(0, train_end - wf_train_sz)
        val_start = train_end + EMBARGO_BARS; val_end = val_start + wf_val_sz
        if val_end > n_wfv: print(f"  Fold {fold+1}: val_end {val_end} > n_wfv {n_wfv} — skipping.", flush=True); break
        n_tr_seqs = train_end - train_start; n_val = wf_val_sz
        if n_val < 200 or n_tr_seqs < 500: print(f"  Fold {fold+1}: seqs insufficient (tr={n_tr_seqs} val={n_val}) — skipping.", flush=True); continue
        mf = hist = pr = ds_tr = ds_vl = sw_tr_w = sw_v_w = sc_fold = fold_feat = fold_gap = None
        try:
            train_raw_end = train_end + SEQ_LEN - 1; val_raw_end = val_end + SEQ_LEN + TB_BARS
            total_raw = min(val_raw_end, len(feat_arr_log)); raw_start = train_start
            fold_feat = feat_arr_log[raw_start:total_raw].copy()
            sc_fold = RobustScaler(); local_tr_raw_end = train_raw_end - raw_start
            fold_feat[:local_tr_raw_end, :N_PRICE] = sc_fold.fit_transform(fold_feat[:local_tr_raw_end, :N_PRICE])
            fold_feat[local_tr_raw_end:, :N_PRICE] = sc_fold.transform(fold_feat[local_tr_raw_end:, :N_PRICE])
            fold_o = o_arr[raw_start:total_raw]; fold_h = h_arr[raw_start:total_raw]
            fold_l = l_arr[raw_start:total_raw]; fold_c = c_arr[raw_start:total_raw]
            fold_atr = atr_arr[raw_start:total_raw]
            fold_gap = gap_arr[raw_start:total_raw] if gap_arr is not None else None
            X_f, y_f, yr_f, at_f, sw_f, hb_f = build_seqs(fold_feat, fold_o, fold_h, fold_l, fold_c, fold_atr, fold_gap, disabled_feats)
            del fold_feat; fold_feat = None; gc.collect()
            local_tr_end = n_tr_seqs; local_val_start = local_tr_end + EMBARGO_BARS; local_val_end = local_val_start + n_val
            if len(X_f) < local_val_end:
                print(f"  Fold {fold+1}: built {len(X_f)} seqs < {local_val_end} needed — skipping.", flush=True)
                del X_f, y_f, yr_f, at_f, sw_f, hb_f, fold_o, fold_h, fold_l, fold_c, fold_atr; gc.collect(); continue
            Xtr_f = X_f[:local_tr_end]; ytr_f = y_f[:local_tr_end]; sw_tr_f = sw_f[:local_tr_end]
            Xv_f = X_f[local_val_start:local_val_end]; yv_f = y_f[local_val_start:local_val_end]
            yr_vf = yr_f[local_val_start:local_val_end]; hb_vf = hb_f[local_val_start:local_val_end]
            sw_vf = sw_f[local_val_start:local_val_end]
            del X_f, y_f, yr_f, at_f, sw_f, hb_f; gc.collect()
            tf.keras.backend.clear_session(); tf.compat.v1.reset_default_graph(); gc.collect()
            sw_tr_w = apply_class_weights(sw_tr_f, ytr_f, cw)
            sw_v_w = apply_class_weights(sw_vf, yv_f, cw)
            with strategy.scope(): mf = build_model(SEQ_LEN, NF, **model_params)
            ds_tr = make_ds(Xtr_f, ytr_f, sw_tr_w, BATCH, shuffle=True)
            ds_vl = make_ds(Xv_f, yv_f, sw_v_w, BATCH)
            es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
            cb = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=0), TerminateOnNaN(), es]
            hist = mf.fit(ds_tr, validation_data=ds_vl, epochs=60, verbose=0, callbacks=cb)
            fold_val_loss = min(hist.history["val_loss"])
            pr = mf.predict(make_ds_eval(Xv_f, BATCH), verbose=0)
            n_a = min(len(pr), len(yr_vf))
            pr_gated = apply_asymmetric_thresholds(pr[:n_a], long_thr=_lthr, short_thr=_sthr)
            bt = backtest(pr_gated, yr_vf[:n_a], hb_vf[:n_a]); wf_metrics.append(bt); fs = _fold_score(bt); fold_scores.append(fs)
            mdd_tag = "🔴" if bt["mdd"] < MAX_MDD else ("⚠️" if bt["mdd"] < MDD_SOFT else "✅")
            print(f"  Fold {fold+1}/{WF_SPLITS} | Ret={bt['ret']:.2%} | WR={bt['wr']:.2%} | L={bt['long_wr']:.2%}({bt['long_trades']}) S={bt['short_wr']:.2%}({bt['short_trades']}) | Sharpe={bt['sharpe']:.2f} | Sortino={bt['sortino']:.2f} | MDD={bt['mdd']:.2%} {mdd_tag} | FoldScore={fs:.4f} | VL={fold_val_loss:.4f}", flush=True)
            del Xtr_f, ytr_f, sw_tr_f, Xv_f, yv_f, sw_vf, yr_vf, hb_vf, fold_o, fold_h, fold_l, fold_c, fold_atr
        finally:
            if mf is not None: del mf
            if hist is not None: del hist
            if pr is not None: del pr
            if ds_tr is not None: del ds_tr
            if ds_vl is not None: del ds_vl
            if sw_tr_w is not None: del sw_tr_w
            if sw_v_w is not None: del sw_v_w
            if sc_fold is not None: del sc_fold
            if fold_feat is not None: del fold_feat
            if fold_gap is not None: del fold_gap
            tf.keras.backend.clear_session(); tf.compat.v1.reset_default_graph(); gc.collect()
    if not wf_metrics: return {}
    agg = {k: float(np.mean([m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    std = {k: float(np.std([m[k] for m in wf_metrics])) for k in wf_metrics[0]}
    print(f"[WFV] mean → Ret={agg['ret']:.2%} | WR={agg['wr']:.2%} | L={agg['long_wr']:.2%} | S={agg['short_wr']:.2%} | Sharpe={agg['sharpe']:.2f} | Sortino={agg['sortino']:.2f} | MDD={agg['mdd']:.2%} | Embargo={EMBARGO_BARS}bars ✓ | FoldScores={[f'{s:.3f}' for s in fold_scores]}", flush=True)
    agg["_std"] = std; agg["_fold_scores"] = fold_scores
    return agg
def print_summary_table():
    if not TRIAL_RESULTS: return
    rows = []
    for tn, r in sorted(TRIAL_RESULTS.items()):
        mdd_tag = "🔴" if r.get("mdd",0) < MAX_MDD else ("⚠️" if r.get("mdd",0) < MDD_SOFT else "✅")
        wr_ok = "✅" if r.get("wr",0) >= TARGET_WR else "  "
        so_ok = "✅" if r.get("sortino",0) >= TARGET_SORTINO else "  "
        rows.append({"T": tn, "Ret": f"{r.get('ret',0):.2%}", "WR": f"{r.get('wr',0):.2%}{wr_ok}", "L_WR": f"{r.get('long_wr',0):.2%}", "S_WR": f"{r.get('short_wr',0):.2%}", "Sortino": f"{r.get('sortino',0):.2f}{so_ok}", "Sharpe": f"{r.get('sharpe',0):.2f}", "MDD": f"{r.get('mdd',0):.2%}{mdd_tag}", "PF": f"{r.get('pf',0):.2f}", "Top5R": f"{r.get('top_ratio',0):.3f}", "NW": f"{r.get('neutral_weight',0):.2f}", "LP": f"{r.get('long_penalty',0):.2f}", "LThr": f"{r.get('long_conf_thr',0):.2f}", "VL": f"{r.get('val_loss',0):.4f}", "Trades": r.get("trades",0), "Score": f"{r.get('score',-99):.4f}"})
    df = pd.DataFrame(rows).set_index("T")
    valid = {t: r for t, r in TRIAL_RESULTS.items() if r.get("mdd",-1) >= MAX_MDD}
    best_tn = max(valid, key=lambda t: valid[t].get("score",-np.inf)) if valid else None
    print("\n" + "="*140); print("TRIAL SUMMARY  v18-Production  (🔴=MDD>25% | ⚠️=MDD>18% | ✅=OK)"); print("="*140)
    print(df.to_string()); print("="*140)
    if best_tn:
        b = TRIAL_RESULTS[best_tn]
        print(f"★ BEST: T{best_tn:02d} | Score={b.get('score',0):.4f} | Ret={b.get('ret',0):.2%} | WR={b.get('wr',0):.2%} | L={b.get('long_wr',0):.2%} | S={b.get('short_wr',0):.2%} | Sortino={b.get('sortino',0):.2f} | MDD={b.get('mdd',0):.2%} | NW={b.get('neutral_weight',0):.2f} | LP={b.get('long_penalty',0):.2f} | LThr={b.get('long_conf_thr',0):.2f} | Top5R={b.get('top_ratio',0):.3f}")
    print("="*140 + "\n", flush=True)
    joblib.dump(df, os.path.join(OUT_DIR, "trial_summary_v18.save"))
def print_stability_report(wfv_metrics):
    print("\n" + "="*90); print("STABILITY REPORT  v18-Production"); print("="*90)
    if FILE_METRICS:
        keys = ["ret","sharpe","sortino","mdd","wr","pf","top_ratio","long_wr","short_wr"]
        print(f"\n{'Metric':<14} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CV%':>8}"); print("-"*62)
        for k in keys:
            vals = np.array([m[k] for m in FILE_METRICS if k in m])
            if len(vals) == 0: continue
            mean_v = vals.mean(); std_v = vals.std(); cv = abs(std_v / (mean_v + EPS)) * 100
            print(f"{k:<14} {mean_v:>10.4f} {std_v:>10.4f} {vals.min():>10.4f} {vals.max():>10.4f} {cv:>7.1f}%")
        print()
    if TRIAL_RESULTS:
        valid = [r for r in TRIAL_RESULTS.values() if r.get("mdd",-1) >= MAX_MDD]
        if valid:
            for k in ["ret","sharpe","sortino","mdd","wr","long_wr","short_wr"]:
                vals = np.array([r.get(k,0) for r in valid])
                print(f"[Trials|{k}]  mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")
    if wfv_metrics and "_std" in wfv_metrics:
        std = wfv_metrics["_std"]
        print(f"\n[WFV fold std] Ret={std.get('ret',0):.2%} | WR={std.get('wr',0):.2%} | Sortino={std.get('sortino',0):.2f} | MDD={std.get('mdd',0):.2%}")
        sortino_ref = abs(wfv_metrics.get("sortino", EPS)); sortino_cv = abs(std.get("sortino",0)) / (sortino_ref + EPS) * 100
        print(f"[WFV Sortino CV] {sortino_cv:.1f}%  {'✅ Consistent' if sortino_cv < 40 else '⚠️  High variance'}")
        fold_scores = wfv_metrics.get("_fold_scores", [])
        if fold_scores:
            arr = np.array(fold_scores); print(f"[WFV FoldScores] {[f'{s:.4f}' for s in fold_scores]}")
            print(f"[WFV Stability]  mean={arr.mean():.4f} | min={arr.min():.4f} | std={arr.std():.4f} | stability={(arr.mean()*0.7 + arr.min()*0.3 - arr.std()*0.5):.4f}")
        print(f"\n[L/S Split WFV]  Long_WR={wfv_metrics.get('long_wr',0):.2%} | Short_WR={wfv_metrics.get('short_wr',0):.2%} | Long_Trades={wfv_metrics.get('long_trades',0):.0f} | Short_Trades={wfv_metrics.get('short_trades',0):.0f} | Long_Ret={wfv_metrics.get('long_ret',0):.4f} | Short_Ret={wfv_metrics.get('short_ret',0):.4f}")
    print("="*90 + "\n", flush=True)
    joblib.dump({"file_metrics": FILE_METRICS, "trial_results": TRIAL_RESULTS, "wfv": wfv_metrics}, os.path.join(OUT_DIR, "stability_report_v18.save"))
def train():
    print(f"[CausalGRU+FiLM-4H-v18] GPUs={NGPU} | SEQ={SEQ_LEN} | NF={NF} | DynFee=clip(FLOOR+ATR×{FEE_ATR_SCALE},FLOOR={DYN_FEE_FLOOR*100:.2f}%,CAP={DYN_FEE_CAP*100:.2f}%) | BATCH={BATCH} | MDD_HARD={MAX_MDD:.0%} | TRIALS={N_TRIALS} | WF_SPLITS={WF_SPLITS} | Embargo={EMBARGO_BARS}bars | GapMask ✓ | TimeNoShift ✓ | SplitLabels ✓ | DynAnn ✓ | AsymGate ✓ | BearRegime ✓", flush=True)
    btc_raw = load_sync_resample(BTC_DIR, "BTC", parquet_path=PARQUET_CACHE_BTC)
    eth_raw = load_sync_resample(ETH_DIR, "ETH", parquet_path=PARQUET_CACHE_ETH)
    btc_raw, eth_raw = sync_assets(btc_raw, eth_raw); gc.collect()
    c_r = btc_raw["close"]; h_r = btc_raw["high"]; l_r = btc_raw["low"]
    tr_r = pd.concat([(h_r - l_r),(h_r - c_r.shift()).abs(),(l_r - c_r.shift()).abs()], axis=1).max(axis=1)
    atr_norm_u = (tr_r.ewm(com=13, adjust=False).mean() / (c_r + EPS)).astype(np.float32)
    feat_df = mkfeat(btc_raw, eth_df=eth_raw, parquet_path=PARQUET_CACHE_FEAT, btc_parquet_path=PARQUET_CACHE_BTC, eth_parquet_path=PARQUET_CACHE_ETH)
    open_s = btc_raw["open"].loc[feat_df.index]; high_s = btc_raw["high"].loc[feat_df.index]
    low_s = btc_raw["low"].loc[feat_df.index]; close_s = btc_raw["close"].loc[feat_df.index]
    atr_s = atr_norm_u.reindex(feat_df.index).ffill().dropna()
    gap_s = btc_raw["is_gap"].reindex(feat_df.index).fillna(0)
    del btc_raw, eth_raw, tr_r, c_r, h_r, l_r, atr_norm_u; gc.collect()
    n_seqs_total = len(feat_df) - SEQ_LEN - TB_BARS + 1; tr_cut, val_cut = _temporal_split(n_seqs_total)
    feat_arr = feat_df.values.astype(np.float32); del feat_df; gc.collect()
    _apply_log_transform(feat_arr)
    feat_arr_log = feat_arr; del feat_arr; gc.collect()
    o_arr = open_s.values.astype(np.float64); h_arr = high_s.values.astype(np.float64)
    l_arr = low_s.values.astype(np.float64); c_arr = close_s.values.astype(np.float64)
    atr_arr = atr_s.values.astype(np.float64); gap_arr = gap_s.values.astype(np.float32)
    del open_s, high_s, low_s, close_s, atr_s, gap_s; gc.collect()
    labels_all, y_raw_all, intrabar_all, hold_bars_all, n_seqs = _build_labels(o_arr, h_arr, l_arr, c_arr, atr_arr)
    sw_all = compute_sample_weights(y_raw_all, intrabar_all)
    sw_all = _apply_gap_mask(sw_all, gap_arr, SEQ_LEN, n_seqs)
    feat_arr_optuna = feat_arr_log.copy()
    train_end_raw = tr_cut + SEQ_LEN - 1; sc_global = fit_scale_inplace(feat_arr_optuna, train_end_raw); gc.collect()
    X_tmp = _build_X_seqs(feat_arr_optuna, n_seqs, disabled_feats=None)
    disabled_feats = select_features(X_tmp[:tr_cut], y_raw_all[:tr_cut])
    if disabled_feats: print(f"[Feature Selection] Disabled {len(disabled_feats)}/{NF} weak features globally.", flush=True)
    del X_tmp; gc.collect()
    X_all = _build_X_seqs(feat_arr_optuna, n_seqs, disabled_feats=disabled_feats)
    del feat_arr_optuna; gc.collect()
    y_all = labels_all.astype(np.int32); yr_all = y_raw_all.astype(np.float32)
    Xtr = X_all[:tr_cut]; Xv = X_all[tr_cut:val_cut]
    ytr = y_all[:tr_cut]; yv = y_all[tr_cut:val_cut]
    sw_tr = sw_all[:tr_cut]; sw_v = sw_all[tr_cut:val_cut]
    yv_raw = yr_all[tr_cut:val_cut]
    hb_v = hold_bars_all[tr_cut:val_cut]
    dist = np.bincount(ytr, minlength=3)
    print(f"[labels] Short={dist[0]:,} | Neutral={dist[1]:,} | Long={dist[2]:,}", flush=True)
    print(f"[seqs]   train={len(Xtr):,} | val={len(Xv):,} | total={n_seqs:,}", flush=True)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=15, multivariate=True), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(lambda t: objective(t, Xtr, ytr, sw_tr, Xv, yv, sw_v, yv_raw, hb_v), n_trials=N_TRIALS, catch=(Exception,), gc_after_trial=True)
    bp = study.best_params; print(f"\n[optuna] best_params={bp}", flush=True)
    print_summary_table()
    tf.keras.backend.clear_session(); tf.compat.v1.reset_default_graph(); gc.collect()
    cw_best = compute_cw(ytr, neutral_weight=bp.get("neutral_weight", 1.5), long_penalty=bp.get("long_penalty", LONG_PENALTY_DEFAULT))
    wfv_params = dict(
        units=bp["units"], nh=bp["nh"], d_rate=bp["d"], lr=bp["lr"],
        noise=bp["noise"], gru_layers=bp.get("gru_layers", 1),
        long_conf_thr=bp.get("long_conf_thr", LONG_CONF_THRESHOLD),
        short_conf_thr=bp.get("short_conf_thr", SHORT_CONF_THRESHOLD),
    )
    wfv_metrics = walk_forward_validate(feat_arr_log, o_arr, h_arr, l_arr, c_arr, atr_arr, gap_arr, val_cut, wfv_params, cw_best, disabled_feats)
    tf.keras.backend.clear_session(); tf.compat.v1.reset_default_graph(); gc.collect()
    apply_scale_transform(feat_arr_log, sc_global)
    X_final = _build_X_seqs(feat_arr_log, n_seqs, disabled_feats=disabled_feats)
    del feat_arr_log; gc.collect()
    _tr_f, _val_f = _temporal_split(n_seqs)
    Xtr_f = X_final[:_tr_f]; Xv_f = X_final[_tr_f:_val_f]; Xte_f = X_final[_val_f:]
    ytr_f = y_all[:_tr_f]; yv_f = y_all[_tr_f:_val_f]
    sw_tr_f = sw_all[:_tr_f]; sw_v_f = sw_all[_tr_f:_val_f]
    yr_te_f = yr_all[_val_f:]; hb_te_f = hold_bars_all[_val_f:]
    del X_final, y_all, yr_all, sw_all, hold_bars_all, labels_all, y_raw_all, intrabar_all; gc.collect()
    sw_tr_w = apply_class_weights(sw_tr_f, ytr_f, cw_best)
    sw_v_w = apply_class_weights(sw_v_f, yv_f, cw_best)
    with strategy.scope(): m = build_model(SEQ_LEN, NF, **{k: v for k, v in wfv_params.items() if k not in ("long_conf_thr", "short_conf_thr")})
    ds_tr = make_ds(Xtr_f, ytr_f, sw_tr_w, BATCH, shuffle=True); ds_vl = make_ds(Xv_f, yv_f, sw_v_w, BATCH)
    es_final = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    cbs = [ModelCheckpoint(os.path.join(OUT_DIR, "model_v18_best_final.keras"), save_best_only=True, monitor="val_loss"), ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1), TerminateOnNaN(), es_final]
    final_hist = m.fit(ds_tr, validation_data=ds_vl, epochs=EPOCHS, verbose=1, callbacks=cbs)
    final_val_loss = min(final_hist.history["val_loss"])
    pr = m.predict(make_ds_eval(Xte_f, BATCH), verbose=0)
    n_min = min(len(pr), len(yr_te_f), len(hb_te_f))
    pr_gated = apply_asymmetric_thresholds(
        pr[:n_min],
        long_thr=bp.get("long_conf_thr", LONG_CONF_THRESHOLD),
        short_thr=bp.get("short_conf_thr", SHORT_CONF_THRESHOLD),
    )
    stats = backtest(pr_gated, yr_te_f[:n_min], hb_te_f[:n_min]); FILE_METRICS.append(stats)
    wfv_fold_scores = wfv_metrics.get("_fold_scores", []) if wfv_metrics else []
    final_score = _composite_score(stats, wfv_scores=wfv_fold_scores, val_loss=final_val_loss)
    print(f"\n[FINAL TEST] Ret={stats['ret']:.2%} | WR={stats['wr']:.2%} | L={stats['long_wr']:.2%}({stats['long_trades']}) | S={stats['short_wr']:.2%}({stats['short_trades']}) | Sharpe={stats['sharpe']:.2f} | Sortino={stats['sortino']:.2f} | MDD={stats['mdd']:.2%} | PF={stats['pf']:.2f} | Trades={stats['trades']} | Top5Ratio={stats['top_ratio']:.3f} | ValLoss={final_val_loss:.4f} | FinalScore={final_score:.4f}", flush=True)
    if wfv_metrics: print(f"[WFV  MEAN]  Ret={wfv_metrics.get('ret',0):.2%} | WR={wfv_metrics.get('wr',0):.2%} | L={wfv_metrics.get('long_wr',0):.2%} | S={wfv_metrics.get('short_wr',0):.2%} | Sortino={wfv_metrics.get('sortino',0):.2f} | Sharpe={wfv_metrics.get('sharpe',0):.2f} | MDD={wfv_metrics.get('mdd',0):.2%}", flush=True)
    print_stability_report(wfv_metrics)
    m.save(os.path.join(OUT_DIR, "model_v18_final.keras")); _try_save_best(m, final_score, label="final")
    joblib.dump({"sc": sc_global, "NF": NF, "N_PRICE": N_PRICE, "ALL_FEATS": ALL_FEATS, "PRICE_FEATS": PRICE_FEATS, "SEQ_LEN": SEQ_LEN, "best_params": bp, "DYN_FEE_FLOOR": DYN_FEE_FLOOR, "DYN_FEE_CAP": DYN_FEE_CAP, "FEE_ATR_SCALE": FEE_ATR_SCALE, "TOP_K_RATIO": TOP_K_RATIO, "REGIME_IDX": REGIME_IDX, "disabled_feats": disabled_feats, "LONG_CONF_THRESHOLD": bp.get("long_conf_thr", LONG_CONF_THRESHOLD), "SHORT_CONF_THRESHOLD": bp.get("short_conf_thr", SHORT_CONF_THRESHOLD)}, os.path.join(OUT_DIR, "scaler_v18.pkl"))
    joblib.dump(stats, os.path.join(OUT_DIR, "stats_v18.save"))
    joblib.dump(wfv_metrics, os.path.join(OUT_DIR, "wfv_metrics_v18.save"))
    joblib.dump(bp, os.path.join(OUT_DIR, "best_params_v18.save"))
    joblib.dump(TRIAL_RESULTS, os.path.join(OUT_DIR, "trial_results_v18.save"))
    del m, ds_tr, ds_vl, final_hist, pr, cbs, es_final, o_arr, h_arr, l_arr, c_arr, atr_arr, gap_arr
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    print(f"[done] → {OUT_DIR}", flush=True)
if __name__ == "__main__":
    train()