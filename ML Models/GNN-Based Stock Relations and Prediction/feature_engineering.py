import os
import pickle
import pandas as pd
import numpy as np
import config

# Paths for saving processed features and targets
FEAT_PKL = os.path.join(config.PROCESSED_DIR, "features.pkl")
TGT_PKL  = os.path.join(config.PROCESSED_DIR, "targets.pkl")

def add_bollinger_bands(close, window=20, k=2):
    ma = close.rolling(window).mean()  # moving average
    std = close.rolling(window).std()  # rolling std dev
    return ma + k*std, ma - k*std    # upper and lower bands

def add_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow            # MACD difference
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # signal line
    return macd_line, signal_line

def prepare_features():
    # Load saved raw stock data
    with open(os.path.join(config.RAW_DIR, "stock_data.pkl"), "rb") as f:
        raw = pickle.load(f)

    feats, tgts = {}, {}
    # Loop over each ticker's raw DataFrame
    for t, df in raw.items():
        # Skip tickers not in our list or missing close prices
        if t not in config.STOCKS or "Close" not in df.columns:
            continue
        df = df.sort_index()           # ensure data is in date order
        close = df["Close"]           # close price series
        volume = df.get("Volume", pd.Series(index=close.index, data=0))
        f = pd.DataFrame(index=close.index)  # feature DataFrame

        # Calculate each configured technical indicator
        for name, w in config.TECHNICAL_INDICATORS.items():
            if name.startswith("returns"):
                f[name] = close.pct_change(w)
            elif name.startswith("ma"):
                f[name] = close.rolling(w, min_periods=1).mean()
            elif name.startswith("volatility"):
                f[name] = close.pct_change().rolling(w, min_periods=1).std()
            elif name.startswith("rsi"):
                # RSI: ratio of average gains to losses
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(w, min_periods=1).mean()
                loss = (-delta).clip(lower=0).rolling(w, min_periods=1).mean()
                rs = gain / loss.replace(0, np.nan)
                f[name] = (100 - (100 / (1 + rs))).fillna(50)
            elif name.startswith("momentum"):
                f[name] = close.diff(w)

        # Add simple moving averages for extra windows
        for w in config.SMA_WINDOWS:
            f[f"sma_{w}"] = close.rolling(w, min_periods=1).mean()

        # Compute Money Flow Index (MFI)
        tp = (df["High"] + df["Low"] + close) / 3  # typical price
        mf = tp * volume                              # money flow
        pos = mf.where(tp.diff() > 0, 0).rolling(config.MFI_WINDOW, min_periods=1).sum()
        neg = mf.where(tp.diff() < 0, 0).rolling(config.MFI_WINDOW, min_periods=1).sum()
        f["mfi"] = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).fillna(50)

        # Add Bollinger Bands and MACD
        bb_u, bb_l = add_bollinger_bands(close)
        macd, sig = add_macd(close)
        f["bb_upper"], f["bb_lower"] = bb_u, bb_l
        f["macd"], f["macd_signal"] = macd, sig

        # On-balance volume (OBV) and volume rate of change
        f["obv"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        f["vroc_5d"] = volume.pct_change(5)

        # Candlestick pattern flags (Doji, bullish engulf)
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        f["is_doji"] = ((c - o).abs() < 0.001 * (h - l)).astype(int)
        f["is_bull_engulf"] = ((c > o.shift()) & (o < c.shift())).astype(int)

        # Clean up infinities and NaNs
        f = f.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        # Label: next-day return > 0 maps to 1 (up) or 0 (down)
        tgt = (close.pct_change(1).shift(-1) > 0).astype(int)
        tgt = tgt.reindex(f.index).fillna(0).astype(int)

        if not f.empty:
            feats[t], tgts[t] = f, tgt

    # Flatten time series: create lookback window of features
    feats_w, tgts_w = {}, {}
    W = config.FEATURE_WINDOW
    for t, df in feats.items():
        if len(df) <= W:
            continue
        # Stack last W days of features into one row
        arr = np.hstack([df.shift(i).iloc[W:].values for i in range(W)])
        cols = [f"{col}_t-{i}" for i in range(W) for col in df.columns]
        feats_w[t] = pd.DataFrame(arr, index=df.index[W:], columns=cols)
        tgts_w[t]  = tgts[t].iloc[W:]

    # Save processed features and targets
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(FEAT_PKL, "wb") as f:
        pickle.dump(feats_w, f)
    with open(TGT_PKL, "wb") as f:
        pickle.dump(tgts_w, f)

    return feats_w, tgts_w