"""
train_model.py
==============
سكريبت لتدريب موديل LSTM على بيانات BTCUSDT من Binance.
- يسحب شمعات 1 دقيقة لآخر 30 يوم  (~43,200 شمعة)
- يحسب المؤشرات: RSI, EMA200, MACD, ATR
- يدرب LSTM بسيط
- يحفظ الموديل في  lstm_model.h5
- يحفظ الـ Scaler في  scaler.save

المتطلبات:
    pip install requests pandas numpy scikit-learn tensorflow joblib
"""

import time
import requests
import numpy as np
import pandas as pd
import joblib

from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════
#  CONFIG  — غيّر اللي تحتاجه هنا بس
# ══════════════════════════════════════════════════════════════════
SYMBOL       = "BTCUSDT"
INTERVAL     = "1m"
DAYS_BACK    = 30          # عدد الأيام اللي هنسحبها
SEQ_LEN      = 60          # طول الـ Sequence اللي الـ LSTM بيشوفها (60 شمعة)
FUTURE_STEPS = 1           # هنتوقع السعر بعد كذا شمعة
BATCH_SIZE   = 64
EPOCHS       = 50
MODEL_PATH   = "lstm_model.h5"
SCALER_PATH  = "scaler.save"
BINANCE_URL  = "https://api.binance.com/api/v3/klines"
LIMIT        = 1000        # الحد الأقصى لكل Request في Binance

# ══════════════════════════════════════════════════════════════════
#  1. سحب البيانات من Binance (Paginated)
# ══════════════════════════════════════════════════════════════════
def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    يسحب شمعات من Binance لمدة `days` يوم.
    بيعمل requests متعددة لأن Binance بيرجع 1000 شمعة بحد أقصى في كل مرة.
    """
    print(f"[+] جاري سحب بيانات {symbol} ({interval}) لآخر {days} يوم ...")

    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_candles = []
    current_start = start_ms
    request_count = 0

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     LIMIT,
        }
        try:
            resp = requests.get(BINANCE_URL, params=params, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
        except requests.RequestException as e:
            print(f"  [!] خطأ في الـ Request: {e}. هنجرب تاني بعد 5 ثواني ...")
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        current_start = candles[-1][0] + 1   # الشمعة التالية بعد آخر واحدة
        request_count += 1

        print(f"  [→] Request #{request_count} | "
              f"إجمالي الشمعات: {len(all_candles):,} | "
              f"آخر تاريخ: {datetime.fromtimestamp(candles[-1][0]/1000).strftime('%Y-%m-%d %H:%M')}")

        time.sleep(0.3)   # تفادي الـ Rate Limit

    print(f"[✓] تم سحب {len(all_candles):,} شمعة بنجاح.\n")

    # ── تحويل لـ DataFrame ───────────────────────────────────────
    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="last")]   # إزالة التكرارات لو فيه
    df.sort_index(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════
#  2. حساب المؤشرات (نفس منطق DataManager)
# ══════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """يضيف RSI, EMA200, MACD, ATR للـ DataFrame."""
    print("[+] جاري حساب المؤشرات ...")
    close = df["close"]

    # ── EMA 200 ──────────────────────────────────────────────────
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # ── RSI 14 ───────────────────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ─────────────────────────────────────────
    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── ATR 14 ───────────────────────────────────────────────────
    hl      = df["high"] - df["low"]
    hcp     = (df["high"] - close.shift()).abs()
    lcp     = (df["low"]  - close.shift()).abs()
    tr      = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()

    # ── Candle Body & Wicks (features إضافية مفيدة) ──────────────
    df["body"]       = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # إزالة الـ NaN الناتجة من المؤشرات
    df.dropna(inplace=True)
    print(f"[✓] تم حساب المؤشرات. الصفوف المتبقية: {len(df):,}\n")
    return df


# ══════════════════════════════════════════════════════════════════
#  3. بناء الـ Sequences للـ LSTM
# ══════════════════════════════════════════════════════════════════
FEATURES = [
    "close", "volume",
    "ema200", "rsi",
    "macd", "macd_signal", "macd_hist",
    "atr", "body", "upper_wick", "lower_wick"
]

def build_sequences(df: pd.DataFrame, seq_len: int, future_steps: int):
    """
    يبني X (sequences) و y (target).
    Target: هل السعر هيرتفع (1) أو يهبط (0) بعد future_steps شمعات.
    """
    print(f"[+] جاري بناء الـ Sequences (seq_len={seq_len}) ...")

    data = df[FEATURES].values   # shape: (n_rows, n_features)

    # ── Scaling ──────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled) - future_steps):
        X.append(data_scaled[i - seq_len : i])          # (seq_len, n_features)

        # Target: 1 = ارتفع السعر، 0 = انخفض
        current_close = data[i - 1, 0]                  # عمود close قبل الـ scaling
        future_close  = data[i + future_steps - 1, 0]
        y.append(1 if future_close > current_close else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[✓] X: {X.shape}  |  y: {y.shape}")
    print(f"    توزيع الـ Target → ارتفع: {y.sum():,.0f} | هبط: {(1-y).sum():,.0f}\n")
    return X, y, scaler


# ══════════════════════════════════════════════════════════════════
#  4. بناء موديل LSTM
# ══════════════════════════════════════════════════════════════════
def build_model(seq_len: int, n_features: int) -> Sequential:
    """بيبني موديل LSTM بسيط وفعّال."""
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(0.2),

        LSTM(64, return_sequences=False),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1,  activation="sigmoid")   # Binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════════
#  5. التدريب والحفظ
# ══════════════════════════════════════════════════════════════════
def train_and_save():
    # ── سحب البيانات ─────────────────────────────────────────────
    df = fetch_klines(SYMBOL, INTERVAL, DAYS_BACK)

    # ── حساب المؤشرات ────────────────────────────────────────────
    df = add_indicators(df)

    # ── بناء الـ Sequences ────────────────────────────────────────
    X, y, scaler = build_sequences(df, SEQ_LEN, FUTURE_STEPS)

    # ── تقسيم Train / Test ───────────────────────────────────────
    # نستخدم آخر 20% كـ test بدون shuffle عشان نحافظ على الترتيب الزمني
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"[+] Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    # ── بناء الموديل ─────────────────────────────────────────────
    model = build_model(SEQ_LEN, X.shape[2])

    # ── Callbacks ────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # ── التدريب ──────────────────────────────────────────────────
    print("[+] بدء التدريب ...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── التقييم ──────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[✓] Test Loss: {loss:.4f}  |  Test Accuracy: {acc*100:.2f}%")

    # ── حفظ الموديل والـ Scaler ───────────────────────────────────
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[✓] تم حفظ الموديل في:  {MODEL_PATH}")
    print(f"[✓] تم حفظ الـ Scaler في: {SCALER_PATH}")
    print("\n🎉 انتهى التدريب بنجاح!")
    return history


# ══════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_and_save()