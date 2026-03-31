import pandas as pd
import numpy as np
import requests
from typing import Optional

class DataManager:
    def __init__(self, symbol: str, interval: str = "1h", limit: int = 500):
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.df: Optional[pd.DataFrame] = None
        
    # ============================================================
    # 1. FETCH KLINES FROM BINANCE
    # ============================================================
    def fetch_klines(self) -> pd.DataFrame:
        """سحب الـ Klines من Binance REST API"""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        raw = response.json()
        
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        # تحويل الأنواع
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        
        self.df = df[["open", "high", "low", "close", "volume"]].copy()
        self._calculate_indicators()
        return self.df
    
    # ============================================================
    # 2. CALCULATE ALL INDICATORS
    # ============================================================
    def _calculate_indicators(self):
        """حساب RSI, EMA200, MACD, ATR على كل الـ DataFrame"""
        df = self.df
        
        # --- EMA 200 ---
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        
        # --- RSI 14 ---
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # --- MACD (12, 26, 9) ---
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # --- ATR 14 ---
        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift()).abs()
        low_cp  = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=13, adjust=False).mean()
        
        self.df = df
    
    # ============================================================
    # 3. UPDATE LAST CANDLE (WebSocket Price)
    # ============================================================
    def update_last_candle(self, new_price: float,
                           new_high: Optional[float] = None,
                           new_low: Optional[float] = None,
                           new_volume: Optional[float] = None) -> dict:
        """
        تحديث آخر صف بالسعر الجديد من WebSocket،
        ثم إعادة حساب المؤشرات للصف الأخير فقط.
        
        Returns: dict بقيم المؤشرات للصف الأخير جاهز للـ AI
        """
        if self.df is None:
            raise RuntimeError("DataFrame is empty. Run fetch_klines() first.")
        
        idx = self.df.index[-1]
        
        # تحديث السعر والـ OHLCV
        self.df.at[idx, "close"] = new_price
        if new_high is not None:
            self.df.at[idx, "high"] = max(self.df.at[idx, "high"], new_high)
        if new_low is not None:
            self.df.at[idx, "low"] = min(self.df.at[idx, "low"], new_low)
        if new_volume is not None:
            self.df.at[idx, "volume"] = new_volume
        
        # ── إعادة حساب المؤشرات للصف الأخير فقط ──────────────────
        self._recalc_last_row()
        
        # إرجاع snapshot للـ AI
        last = self.df.iloc[-1]
        return {
            "symbol":      self.symbol,
            "close":       round(last["close"], 6),
            "ema200":      round(last["ema200"], 6),
            "rsi":         round(last["rsi"], 2),
            "macd":        round(last["macd"], 6),
            "macd_signal": round(last["macd_signal"], 6),
            "macd_hist":   round(last["macd_hist"], 6),
            "atr":         round(last["atr"], 6),
        }
    
    # ============================================================
    # 4. RECALCULATE LAST ROW ONLY (Fast Path)
    # ============================================================
    def _recalc_last_row(self):
        """
        إعادة حساب المؤشرات للصف الأخير فقط بدون إعادة حساب كل الـ DataFrame.
        يستخدم القيم السابقة (n-1) كـ 'state' لحساب EWM incremental.
        """
        df   = self.df
        last = df.index[-1]
        prev = df.index[-2]
        
        close_last = df.at[last, "close"]
        close_prev = df.at[prev, "close"]
        
        # ── EMA 200 ──────────────────────────────────────────────
        alpha_200 = 2 / (200 + 1)
        df.at[last, "ema200"] = (
            close_last * alpha_200 + df.at[prev, "ema200"] * (1 - alpha_200)
        )
        
        # ── RSI ───────────────────────────────────────────────────
        alpha_rsi = 1 / 14          # com=13  →  alpha = 1/(1+13)
        delta = close_last - close_prev
        gain  = max(delta, 0.0)
        loss  = max(-delta, 0.0)
        
        # نحتاج avg_gain/avg_loss من الصف السابق
        rsi_prev = df.at[prev, "rsi"]
        # استخرج avg_gain & avg_loss من rsi_prev + close series
        # نحسبها مرة من الـ full series للـ prev row (cached in _last_rsi_state)
        avg_gain_prev, avg_loss_prev = self._get_rsi_state(prev)
        
        avg_gain = avg_gain_prev * (1 - alpha_rsi) + gain * alpha_rsi
        avg_loss = avg_loss_prev * (1 - alpha_rsi) + loss * alpha_rsi
        
        rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
        df.at[last, "rsi"] = 100 - (100 / (1 + rs)) if not np.isnan(rs) else 50.0
        
        # حفظ الـ state الجديد
        self._rsi_state = (avg_gain, avg_loss)
        
        # ── MACD ──────────────────────────────────────────────────
        alpha12 = 2 / (12 + 1)
        alpha26 = 2 / (26 + 1)
        alpha9  = 2 / (9 + 1)
        
        ema12_prev = df.at[prev, "macd"] + df["close"].ewm(span=26, adjust=False).mean().iloc[-2]
        # أبسط: نحتفظ بـ ema12/ema26 كـ state
        ema12_prev_val, ema26_prev_val = self._get_macd_state(prev)
        
        ema12 = close_last * alpha12 + ema12_prev_val * (1 - alpha12)
        ema26 = close_last * alpha26 + ema26_prev_val * (1 - alpha26)
        macd  = ema12 - ema26
        
        signal_prev = df.at[prev, "macd_signal"]
        signal = macd * alpha9 + signal_prev * (1 - alpha9)
        
        df.at[last, "macd"]        = macd
        df.at[last, "macd_signal"] = signal
        df.at[last, "macd_hist"]   = macd - signal
        
        self._macd_state = (ema12, ema26)
        
        # ── ATR ───────────────────────────────────────────────────
        alpha_atr = 1 / 14
        high_last  = df.at[last, "high"]
        low_last   = df.at[last, "low"]
        close_prev = df.at[prev, "close"]
        
        tr = max(
            high_last - low_last,
            abs(high_last - close_prev),
            abs(low_last  - close_prev)
        )
        atr_prev = df.at[prev, "atr"]
        df.at[last, "atr"] = atr_prev * (1 - alpha_atr) + tr * alpha_atr
    
    # ── State Helpers ─────────────────────────────────────────────
    def _get_rsi_state(self, idx) -> tuple:
        """إرجاع (avg_gain, avg_loss) للـ index المطلوب"""
        if hasattr(self, "_rsi_state"):
            return self._rsi_state
        # حساب لأول مرة من الـ full series
        delta    = self.df["close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        pos = self.df.index.get_loc(idx)
        self._rsi_state = (avg_gain.iloc[pos], avg_loss.iloc[pos])
        return self._rsi_state
    
    def _get_macd_state(self, idx) -> tuple:
        """إرجاع (ema12, ema26) للـ index المطلوب"""
        if hasattr(self, "_macd_state"):
            return self._macd_state
        ema12 = self.df["close"].ewm(span=12, adjust=False).mean()
        ema26 = self.df["close"].ewm(span=26, adjust=False).mean()
        pos   = self.df.index.get_loc(idx)
        self._macd_state = (ema12.iloc[pos], ema26.iloc[pos])
        return self._macd_state