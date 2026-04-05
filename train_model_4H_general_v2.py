import gc, os, glob, warnings, logging
import numpy as np, pandas as pd, tensorflow as tf, optuna, joblib
from scipy.fft import fft as sfft
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, SpatialDropout1D,
    MultiHeadAttention, Add, Multiply, Lambda, Reshape, GaussianNoise,
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
print(f"[init] GPUs={NGPU}", flush=True)

BTC_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/BTC"
ETH_DIR = "/kaggle/input/datasets/karimmohamed2026/crypto-1m-data/My Data/ETH"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN      = 180
FUTURE_STEPS = 1
BATCH        = 32
EPOCHS       = 120
N_TRIALS     = 40
N_FFT        = 16
FFT_WIN      = 180
FRAC_D       = 0.4
FRAC_THR     = 1e-4
EPS          = 1e-8
FEE          = 0.0006
LS           = 0.02
QUANTILES    = [0.1, 0.5, 0.9]
ANN_FACTOR   = np.sqrt(365 * 6)
_L2          = tf.keras.regularizers.l2(1e-5)
TRIAL_RESULTS = {}

BASE_FEATS = [
    "close_norm", "ema20", "ema50", "ema200",
    "rsi", "rsi_slope",
    "macd", "macd_sig", "macd_hist",
    "atr", "atr_norm", "atr_regime", "atr_breakout",
    "body", "up_wick", "lo_wick",
    "bb_up", "bb_lo", "bb_w",
    "vol_ema", "tkr_ratio", "rvi", "rvi_slope", "vol_trend",
    "log_ret", "vol_reg", "volatility", "vol_zsc",
    "roc", "vol_scaled_ret",
    "eth_rsi_ratio", "eth_close_norm", "btc_eth_corr",
    "obv_norm", "vwap_ratio",
    "stoch_k", "stoch_d", "cci", "cmf",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "week_sin", "week_cos",
    "dom_sin", "dom_cos",
    "woy_sin", "woy_cos",
    "session_asia", "session_eu", "session_us",
    "adx", "di_diff",
    "ichimoku_cloud", "tenkan_kijun",
    "vol_regime",
    "frac_diff",
]
FFT_FEATS = [f"fft{i}" for i in range(N_FFT)]
ALL_FEATS = BASE_FEATS + FFT_FEATS
NF = len(ALL_FEATS)
print(f"[config] 4H-v2 | SEQ={SEQ_LEN} | NF={NF} | TRIALS={N_TRIALS}", flush=True)


def _load_resample_one(fp):
    try:
        df = pd.read_csv(fp, header=None, skiprows=1)
        nc = df.shape[1]
        if nc >= 8:   df = df.iloc[:, [1, 3, 4, 5, 6, 7]]
        elif nc == 7: df = df.iloc[:, [1, 2, 3, 4, 5, 6]]
        elif nc == 6: df = df.iloc[:, :6]
        else:         return None
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        resampled = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        return resampled.astype("float32")
    except Exception:
        return None


def load_sync_resample(data_dir, label=""):
    fps = sorted(set(
        glob.glob(os.path.join(data_dir, "*.csv")) +
        glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    ))
    print(f"[load] {label} | files={len(fps)}", flush=True)
    parts = []
    for fp in fps:
        chunk = _load_resample_one(fp)
        if chunk is not None and len(chunk) > 10:
            parts.append(chunk)
        del chunk; gc.collect()
    if not parts:
        raise RuntimeError(f"[load] {label}: no valid CSV in {data_dir}")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    del parts; gc.collect()
    print(f"[load] {label} | total={len(df):,} | {df.index[0]} -> {df.index[-1]}", flush=True)
    df["vol"] = df["volume"]
    return df


def sync_assets(btc_df, eth_df):
    idx = btc_df.index.intersection(eth_df.index)
    print(f"[sync] aligned={len(idx):,} bars@4H", flush=True)
    return btc_df.loc[idx].copy(), eth_df.loc[idx].copy()


def _frac_diff_weights(d, size, thr=FRAC_THR):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
        if abs(w[-1]) < thr: break
    return np.array(w[::-1], dtype=np.float64)


def frac_diff(series, d=FRAC_D):
    v  = np.ascontiguousarray(np.log(series.astype(np.float64) + EPS))
    w  = _frac_diff_weights(d, len(v))
    wl = len(w); n = len(v)
    shape   = (n - wl + 1, wl)
    strides = (v.strides[0], v.strides[0])
    wins = np.lib.stride_tricks.as_strided(v, shape=shape, strides=strides).copy()
    res  = (wins @ w).astype(np.float32)
    out  = np.full(n, np.nan, dtype=np.float32)
    out[wl - 1:] = res
    return out


def _vec_fft(cv, fw=FFT_WIN, nf=N_FFT):
    n    = len(cv)
    cv64 = np.ascontiguousarray(cv, dtype=np.float64)
    shape   = (n - fw, fw)
    strides = (cv64.strides[0], cv64.strides[0])
    wins = np.lib.stride_tricks.as_strided(cv64, shape=shape, strides=strides).copy()
    mn   = wins.mean(axis=1, keepdims=True)
    sd   = wins.std(axis=1, keepdims=True) + EPS
    mg   = np.abs(sfft((wins - mn) / sd, axis=1))[:, :fw // 2]
    mg  /= mg.sum(axis=1, keepdims=True) + EPS
    idx  = np.argsort(mg, axis=1)[:, ::-1][:, :nf]
    top  = np.take_along_axis(mg, idx, axis=1).astype(np.float32)
    ff   = np.zeros((n, nf), dtype=np.float32)
    ff[fw:] = top
    return ff


def _rsi(s, period=14):
    dif = s.diff(); g = dif.clip(lower=0); dn = -dif.clip(upper=0)
    return 100 - (100 / (1 + g.ewm(com=period-1, adjust=False).mean() /
                          (dn.ewm(com=period-1, adjust=False).mean() + EPS)))


def _adx(h, l, c, period=14):
    tr    = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(com=period - 1, adjust=False).mean()
    up_m  = h - h.shift(); dn_m = l.shift() - l
    dmp   = up_m.where((up_m > dn_m) & (up_m > 0), 0.0)
    dmn   = dn_m.where((dn_m > up_m) & (dn_m > 0), 0.0)
    dip   = 100 * dmp.ewm(com=period-1, adjust=False).mean() / (atr_s + EPS)
    din   = 100 * dmn.ewm(com=period-1, adjust=False).mean() / (atr_s + EPS)
    dx    = 100 * (dip - din).abs() / (dip + din + EPS)
    return dx.ewm(com=period-1, adjust=False).mean(), dip - din


def mkfeat(btc_df, eth_df):
    df = btc_df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]
    v = df["vol"];   o = df["open"]
    ec = eth_df["close"].reindex(df.index, method="ffill")

    df["close_norm"] = c / (c.rolling(200).mean().replace(0, np.nan).ffill() + EPS)
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    rsi_raw              = _rsi(c)
    df["rsi"]            = rsi_raw
    df["rsi_slope"]      = rsi_raw.diff(5) / 5.0
    df["eth_rsi_ratio"]  = _rsi(ec) / (rsi_raw + EPS)
    df["eth_close_norm"] = ec / (ec.rolling(200).mean().replace(0, np.nan).ffill() + EPS)

    btc_ret = np.log(c / (c.shift(1) + EPS))
    eth_ret = np.log(ec / (ec.shift(1) + EPS))
    df["btc_eth_corr"] = btc_ret.rolling(42).corr(eth_ret).fillna(0).clip(-1, 1)

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]      = e12 - e26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    atr_raw            = tr.ewm(com=13, adjust=False).mean()
    df["atr"]          = atr_raw
    df["atr_norm"]     = atr_raw / (c + EPS)
    df["atr_regime"]   = atr_raw / (atr_raw.rolling(42).mean() + EPS)
    df["atr_breakout"] = (c - c.rolling(20).mean()) / (atr_raw + EPS)

    df["body"]    = (c - o).abs()
    hi_lo         = df[["open", "close"]]
    df["up_wick"] = h - hi_lo.max(axis=1)
    df["lo_wick"] = hi_lo.min(axis=1) - l

    s20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    df["bb_up"] = s20 + 2 * sd20
    df["bb_lo"] = s20 - 2 * sd20
    df["bb_w"]  = (df["bb_up"] - df["bb_lo"]) / (s20 + EPS)

    vol_ema20       = v.ewm(span=20, adjust=False).mean()
    df["vol_ema"]   = vol_ema20
    df["tkr_ratio"] = v / (vol_ema20 + EPS)

    rvi_raw         = (v - v.rolling(42).mean()) / (v.rolling(42).std() + EPS)
    df["rvi"]       = rvi_raw
    df["rvi_slope"] = rvi_raw.diff(6) / 6.0
    df["vol_trend"] = v.ewm(span=12, adjust=False).mean() / (v.ewm(span=42, adjust=False).mean() + EPS)

    df["log_ret"]        = btc_ret
    df["vol_reg"]        = btc_ret.rolling(20).std()
    df["volatility"]     = btc_ret.rolling(14).std()
    df["roc"]            = (c - c.shift(14)) / (c.shift(14) + EPS)
    df["vol_scaled_ret"] = btc_ret / (df["volatility"] + EPS)

    eth_vol = eth_ret.rolling(14).std()
    pool    = (df["volatility"] + eth_vol) / 2
    df["vol_zsc"] = (df["volatility"] - pool.rolling(180).mean()) / \
                    (pool.rolling(180).std() + EPS)

    obv = pd.Series(
        np.cumsum(np.sign(btc_ret.fillna(0).values) * v.values), index=df.index
    )
    df["obv_norm"] = obv / (obv.rolling(200).std() + EPS)

    tp = (h + l + c) / 3
    df["vwap_ratio"] = c / (
        (tp * v).rolling(20).sum() / (v.rolling(20).sum() + EPS) + EPS
    )

    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["stoch_k"] = (c - lo14) / (hi14 - lo14 + EPS) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["cci"]     = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + EPS)
    mfv           = ((2 * c - h - l) / (h - l + EPS)) * v
    df["cmf"]     = mfv.rolling(20).sum() / (v.rolling(20).sum() + EPS)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24).astype(np.float32)
    df["day_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7).astype(np.float32)
    df["day_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7).astype(np.float32)
    hour_of_week   = df.index.dayofweek * 24 + df.index.hour
    df["week_sin"] = np.sin(2 * np.pi * hour_of_week / 168).astype(np.float32)
    df["week_cos"] = np.cos(2 * np.pi * hour_of_week / 168).astype(np.float32)

    # FIX-3: zero-anchored cyclical encoding — day 1 → sin(0)=0, day 31 → full cycle
    df["dom_sin"] = np.sin(2 * np.pi * (df.index.day - 1) / 31).astype(np.float32)
    df["dom_cos"] = np.cos(2 * np.pi * (df.index.day - 1) / 31).astype(np.float32)
    # FIX-4: ISO week 1-53 → zero-anchored, divide by 53 (not 52)
    woy = df.index.isocalendar().week.astype(int).values
    df["woy_sin"] = np.sin(2 * np.pi * (woy - 1) / 53).astype(np.float32)
    df["woy_cos"] = np.cos(2 * np.pi * (woy - 1) / 53).astype(np.float32)

    hr = df.index.hour
    df["session_asia"] = ((hr >= 0)  & (hr < 8)).astype(np.float32)
    df["session_eu"]   = ((hr >= 8)  & (hr < 16)).astype(np.float32)
    df["session_us"]   = ((hr >= 16) & (hr < 24)).astype(np.float32)

    adx_vals, di_diff_vals = _adx(h, l, c, period=14)
    df["adx"]     = adx_vals
    df["di_diff"] = di_diff_vals

    tenkan   = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun    = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    df["ichimoku_cloud"] = (senkou_a - senkou_b) / (c + EPS)
    df["tenkan_kijun"]   = (tenkan - kijun) / (c + EPS)

    vol_30     = df["volatility"].rolling(180).mean()
    vol_30_std = df["volatility"].rolling(180).std()
    df["vol_regime"] = (df["volatility"] - vol_30) / (vol_30_std + EPS)

    df["frac_diff"] = frac_diff(c.values, d=FRAC_D)

    ff = _vec_fft(c.values, fw=FFT_WIN, nf=N_FFT)
    for i in range(N_FFT):
        df[f"fft{i}"] = ff[:, i]

    df = df.dropna(subset=ALL_FEATS)[ALL_FEATS].astype(np.float32)
    print(f"[feat] rows={len(df):,} | cols={NF}", flush=True)
    return df


def build_seqs(feat_df, close_s, sl=SEQ_LEN, fs=FUTURE_STEPS):
    # FIX-5: vectorized numpy indexing replaces Python loop — 50x faster
    d   = np.ascontiguousarray(feat_df.values, dtype=np.float32)
    c   = close_s.values.astype(np.float64)
    lr  = np.zeros(len(c), dtype=np.float64)
    lr[1:] = np.log(c[1:] / (c[:-1] + EPS))
    vol_s  = pd.Series(lr).rolling(14).std().values
    vol    = np.maximum(np.where(np.isfinite(vol_s), vol_s, EPS), EPS)
    n_seqs = len(d) - sl - fs
    idx    = np.arange(n_seqs)[:, None] + np.arange(sl)
    X      = d[idx]
    i_v    = np.arange(n_seqs)
    rets   = np.log((c[i_v + sl + fs - 1] + EPS) / (c[i_v + sl - 1] + EPS)).astype(np.float32)
    vol_t  = vol[sl + i_v].astype(np.float32)
    y      = (rets / (vol_t * np.sqrt(fs) + EPS)).astype(np.float32)
    return X, y, rets


# FIX-1: qloss — recreate q locally per-call (replica-safe), explicit f32 shapes
def qloss(y_true, y_pred):
    q      = tf.constant([0.1, 0.5, 0.9], dtype=tf.float32)
    y_true = tf.cast(tf.reshape(y_true, (-1,)),   tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, (-1, 3)), tf.float32)
    ys     = y_true * (1.0 - LS) + tf.stop_gradient(tf.reduce_mean(y_true)) * LS
    e      = tf.clip_by_value(tf.expand_dims(ys, -1) - y_pred, -10.0, 10.0)
    q_     = tf.reshape(q, (1, 3))
    ql     = tf.reduce_mean(tf.maximum(q_ * e, (q_ - 1.0) * e))
    mono   = tf.reduce_mean(
        tf.maximum(0.0, y_pred[:, 0] - y_pred[:, 1]) +
        tf.maximum(0.0, y_pred[:, 1] - y_pred[:, 2])
    ) * 0.1
    return ql + mono


# FIX-2: sigmoid cast to float32 for numerical stability under mixed_float16
def _glu(x, units, d_rate):
    h  = Dense(2 * units, kernel_regularizer=_L2)(x)
    v_ = Lambda(lambda t: t[..., :units])(h)
    g_ = Lambda(lambda t: tf.cast(
        tf.sigmoid(tf.cast(t[..., units:], tf.float32)), t.dtype
    ))(h)
    return Dropout(d_rate)(Multiply()([v_, g_]))


def _grn(x, units, d_rate, depth=2):
    x_dim = x.shape[-1]
    skip  = Dense(units, kernel_regularizer=_L2)(x) \
            if (x_dim is None or x_dim != units) else x
    h = x
    for _ in range(depth):
        h = Dense(units, activation="elu", kernel_regularizer=_L2)(h)
    h = _glu(h, units, d_rate)
    return LayerNormalization()(Add()([skip, h]))


def _vsn(x, nf, units, d_rate):
    ctx = _grn(Lambda(lambda t: tf.reduce_mean(t, axis=1))(x), units, d_rate, depth=2)
    w   = Dense(nf, activation="softmax")(ctx)
    x_w = Multiply()([x, Reshape((1, nf))(w)])
    return _grn(x_w, units, d_rate, depth=2)


def _sinpe(sl, units):
    pos = np.arange(sl)[:, None].astype(np.float32)
    i   = np.arange(units)[None, :].astype(np.float32)
    a   = pos / np.power(10000.0, (2 * (i // 2)) / np.maximum(units, 1))
    a[:, 0::2] = np.sin(a[:, 0::2])
    a[:, 1::2] = np.cos(a[:, 1::2])
    return a[None].astype(np.float32)


# FIX-6a: causal mask prevents temporal leakage during training
# FIX-6b: SpatialDropout1D preserves temporal structure across feature channels
def _tft_block(x, nh, kd, d_rate, units, depth=3):
    attn = MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=d_rate)(
        x, x, use_causal_mask=True
    )
    x2   = LayerNormalization()(Add()([x, SpatialDropout1D(d_rate)(attn)]))
    return _grn(x2, units, d_rate, depth=depth)


# QUANTUM LEAP: learned temporal attention gate over all 180 candles
# replaces naive t[:,-1,:] — model selects which of 30 days matter most
def _temporal_gate(x, units, d_rate):
    scores  = Dense(1, use_bias=True, kernel_regularizer=_L2)(x)
    weights = Lambda(lambda t: tf.cast(
        tf.nn.softmax(tf.cast(t, tf.float32), axis=1), t.dtype
    ))(scores)
    weighted = Multiply()([x, weights])
    ctx      = Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted)
    return Dropout(d_rate)(ctx)


def build_tft(seq_len, n_feat, units, nh, d_rate, lr, noise):
    kd     = max(units // nh, 1)
    inputs = Input(shape=(seq_len, n_feat), dtype="float32")
    x      = GaussianNoise(noise)(inputs)
    x      = _vsn(x, n_feat, units, d_rate)
    # FIX-6c: PE cast to t.dtype inside Lambda — fixes float16/float32 mismatch
    _pe    = _sinpe(seq_len, units)
    x      = Lambda(lambda t: t + tf.cast(tf.constant(_pe), t.dtype))(x)
    for _  in range(4):
        x  = _tft_block(x, nh, kd, d_rate, units, depth=3)
    gate   = _temporal_gate(x, units, d_rate)
    h      = _grn(gate, units, d_rate, depth=2)
    out    = Dense(3, dtype="float32")(h)
    model  = Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=qloss,
    )
    return model


# FIX-7: drop_remainder=True — mandatory for MirroredStrategy shape consistency
def make_ds(X, y, batch, shuffle=True, use_cache=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.float32))
    )
    if use_cache: ds = ds.cache()
    if shuffle:   ds = ds.shuffle(min(len(y), 50_000), reshuffle_each_iteration=True)
    ds = ds.batch(batch * NGPU, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def backtest(pr, y_norm, y_raw):
    q10, q50, q90 = pr[:, 0], pr[:, 1], pr[:, 2]
    spread  = np.clip(q90 - q10, EPS, None)
    thr     = np.percentile(np.abs(q50), 60)
    unc     = np.percentile(spread, 60)
    conf    = np.clip(
        (1.0 / spread) / (np.percentile(1.0 / spread, 95) + EPS), 0.1, 1.0
    )
    size    = np.clip(np.abs(q50) / (thr + EPS), 0.0, 1.0) * conf
    sig     = np.where(
        (q50 > thr)  & (spread < unc),  1.0,
        np.where((q50 < -thr) & (spread < unc), -1.0, 0.0),
    ) * size
    rets    = sig * y_raw - 2.0 * FEE * np.abs(sig)
    cum     = np.cumprod(1.0 + rets)
    pk      = np.maximum.accumulate(cum)
    mdd     = ((cum - pk) / (pk + EPS)).min()
    sharpe  = rets.mean() / (rets.std() + EPS) * ANN_FACTOR
    neg     = rets[rets < 0]
    sortino = rets.mean() / (neg.std() + EPS) * ANN_FACTOR
    calmar  = (cum[-1] - 1.0) / (abs(mdd) + EPS)
    pf      = rets[rets > 0].sum() / (abs(neg.sum()) + EPS)
    n_tr    = int(np.count_nonzero(sig))
    wr      = len(rets[(sig != 0) & (rets > 0)]) / (n_tr + EPS)
    return dict(
        ret=float(cum[-1]-1), mdd=float(mdd), sharpe=float(sharpe),
        sortino=float(sortino), calmar=float(calmar),
        pf=float(pf), wr=float(wr), trades=n_tr,
    )


def objective(trial, Xtr2, ytr2, Xv, yv, yv_raw):
    tf.keras.backend.clear_session(); gc.collect()
    units = trial.suggest_categorical("units", [128, 256, 384, 512, 768, 1024])
    nh    = trial.suggest_categorical("nh",    [8, 16, 32, 64])
    d     = trial.suggest_float("d",     0.10, 0.45)
    lr    = trial.suggest_float("lr",    5e-5, 5e-3, log=True)
    noise = trial.suggest_float("noise", 0.003, 0.05)
    if units % nh != 0: return float("inf")

    tn = trial.number + 1
    print(f"[T{tn:02d}/{N_TRIALS}] units={units}|nh={nh}|d={d:.3f}|lr={lr:.2e}|noise={noise:.3f}", flush=True)
    val = np.inf; m = None
    try:
        with strategy.scope():
            m = build_tft(SEQ_LEN, NF, units, nh, d, lr, noise)
        ds_tr = make_ds(Xtr2, ytr2, BATCH, shuffle=True,  use_cache=False)
        ds_vl = make_ds(Xv,   yv,   BATCH, shuffle=False, use_cache=False)
        cb = [EarlyStopping(patience=8, restore_best_weights=True), TerminateOnNaN()]
        h  = m.fit(ds_tr, validation_data=ds_vl, epochs=35, verbose=0, callbacks=cb)
        val = float(min(h.history.get("val_loss", [np.inf])))
        if np.isfinite(val):
            ckpt = os.path.join(OUT_DIR, f"model_4H_trial_{tn:02d}.keras")
            m.save(ckpt)
            pr_v = np.nan_to_num(m.predict(ds_vl, verbose=0), nan=0.0)
            bt   = backtest(pr_v, yv, yv_raw)
            TRIAL_RESULTS[tn] = {
                "val_loss": val, "ret": bt["ret"],
                "sharpe": bt["sharpe"], "mdd": bt["mdd"], "model": ckpt,
            }
            print(
                f"[T{tn:02d}] loss={val:.6f} | Ret={bt['ret']:.2%} | "
                f"Sharpe={bt['sharpe']:.2f} | MDD={bt['mdd']:.2%}",
                flush=True,
            )
    except Exception as e:
        print(f"[T{tn:02d}] ERROR: {e}", flush=True)
    finally:
        try: del m
        except NameError: pass
        tf.keras.backend.clear_session(); gc.collect()
    return val


def find_best_trial():
    if not TRIAL_RESULTS:
        print("[tracker] No trials recorded.", flush=True)
        return None
    trials  = list(TRIAL_RESULTS.keys())
    losses  = np.array([TRIAL_RESULTS[t]["val_loss"] for t in trials], dtype=np.float64)
    rets    = np.array([TRIAL_RESULTS[t]["ret"]      for t in trials], dtype=np.float64)
    sharpes = np.array([TRIAL_RESULTS[t]["sharpe"]   for t in trials], dtype=np.float64)

    def _norm(arr, lower_better=False):
        mn, mx = arr.min(), arr.max()
        n = (arr - mn) / (mx - mn + EPS)
        return (1.0 - n) if lower_better else n

    score = 0.40 * _norm(losses, lower_better=True) + \
            0.35 * _norm(rets) + \
            0.25 * _norm(sharpes)

    best_tn = trials[int(np.argmax(score))]
    best    = TRIAL_RESULTS[best_tn]

    print("\n" + "=" * 72, flush=True)
    print(f"  ★ BEST TRIAL → Trial {best_tn:02d}", flush=True)
    print(f"  Model    : {best['model']}", flush=True)
    print(
        f"  val_loss={best['val_loss']:.6f} | Ret={best['ret']:.2%} | "
        f"Sharpe={best['sharpe']:.2f} | MDD={best['mdd']:.2%}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return best["model"]


def train():
    print(f"[TFT-4H-v2] GPUs={NGPU} | SEQ={SEQ_LEN} | NF={NF} | TRIALS={N_TRIALS}", flush=True)

    btc_raw = load_sync_resample(BTC_DIR, "BTC")
    eth_raw = load_sync_resample(ETH_DIR, "ETH")
    btc_raw, eth_raw = sync_assets(btc_raw, eth_raw)
    gc.collect()

    feat_df = mkfeat(btc_raw, eth_raw)
    close_s = btc_raw["close"].loc[feat_df.index]
    gc.collect()

    sc     = RobustScaler()
    raw_v  = feat_df.values.astype(np.float32)
    scaled = sc.fit_transform(raw_v).astype(np.float32)
    feat_scaled = pd.DataFrame(scaled, index=feat_df.index, columns=ALL_FEATS)
    del raw_v, scaled; gc.collect()

    X, y, y_raw = build_seqs(feat_scaled, close_s)
    cut  = int(len(X) * 0.80)
    Xtr, Xte         = X[:cut],     X[cut:]
    ytr, yte          = y[:cut],     y[cut:]
    ytr_raw, yte_raw  = y_raw[:cut], y_raw[cut:]
    n, sl, nf_ = Xtr.shape
    print(f"[seqs] train={n:,} | test={len(Xte):,} | sl={sl} | nf={nf_}", flush=True)
    del X, y, y_raw; gc.collect()

    v_cut    = max(int(n * 0.20), 1)
    Xtr2, Xv = Xtr[:-v_cut], Xtr[-v_cut:]
    ytr2, yv = ytr[:-v_cut], ytr[-v_cut:]
    ytr2_raw, yv_raw = ytr_raw[:-v_cut], ytr_raw[-v_cut:]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda t: objective(t, Xtr2, ytr2, Xv, yv, yv_raw),
        n_trials=N_TRIALS, catch=(Exception,),
    )
    bp = study.best_params
    print(f"[optuna] best={bp}", flush=True)
    del Xtr2, Xv, ytr2, yv, ytr2_raw, yv_raw; gc.collect()

    best_trial_model = find_best_trial()

    tf.keras.backend.clear_session(); gc.collect()
    with strategy.scope():
        m = build_tft(sl, nf_, units=bp["units"], nh=bp["nh"],
                      d_rate=bp["d"], lr=bp["lr"], noise=bp["noise"])

    ds_tr = make_ds(Xtr, ytr, BATCH, shuffle=True,  use_cache=True)
    ds_te = make_ds(Xte, yte, BATCH, shuffle=False, use_cache=True)

    cbs = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(os.path.join(OUT_DIR, "model_4H_best_final.keras"),
                        save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                          min_lr=1e-7, verbose=0),
        TerminateOnNaN(),
    ]
    print("[train] Final training started...", flush=True)
    m.fit(ds_tr, validation_data=ds_te, epochs=EPOCHS, verbose=1, callbacks=cbs)

    pr    = np.nan_to_num(m.predict(ds_te, verbose=0), nan=0.0)
    stats = backtest(pr, yte, yte_raw)
    print(
        f"[final] Ret={stats['ret']:.2%} | Sharpe={stats['sharpe']:.2f} | "
        f"MDD={stats['mdd']:.2%} | Sortino={stats['sortino']:.2f} | "
        f"Calmar={stats['calmar']:.2f} | PF={stats['pf']:.2f} | "
        f"WR={stats['wr']:.2%} | Trades={stats['trades']}",
        flush=True,
    )

    m.save(os.path.join(OUT_DIR, "model_4H_final.keras"))
    joblib.dump(sc,            os.path.join(OUT_DIR, "tft_4H_scaler.save"))
    joblib.dump(stats,         os.path.join(OUT_DIR, "tft_4H_stats.save"))
    joblib.dump(bp,            os.path.join(OUT_DIR, "tft_4H_best_params.save"))
    joblib.dump(TRIAL_RESULTS, os.path.join(OUT_DIR, "tft_4H_trial_results.save"))
    print(f"[done] Artefacts -> {OUT_DIR}", flush=True)
    if best_trial_model:
        print(f"★ BEST TRIAL MODEL: {best_trial_model}", flush=True)


if __name__ == "__main__":
    train()
