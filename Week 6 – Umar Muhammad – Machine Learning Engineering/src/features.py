"""Feature engineering for next-trading-day closing price prediction.

Row t describes trading day t. EVERY feature uses only data up to the close of day t-1 (never day t):
  * 'previous day' prices and volume are the day t-1 values,
  * moving averages and rolling statistics are computed on a series that is shifted by one day first,
so nothing about day t (the day being predicted) can leak into its own features.
Targets: `close` (price) and `target_return_pct` = (close_t / close_{t-1} - 1) * 100.
The same function builds the training table and the API's single-row input (one code path, no train/serve skew).
"""
import numpy as np
import pandas as pd

HISTORY_DAYS = 20      # trading days of history needed to build the features of one new day

REQUIRED_FEATURES = ["prev_close", "prev_open", "prev_high", "prev_low", "prev_volume",
                     "ma_5", "ma_10", "ma_20", "prev_return_pct"]
RATIO_FEATURES = ["prev_return_pct", "prev_open_gap_pct", "prev_range_pct", "close_vs_ma5_pct", "close_vs_ma10_pct",
                  "close_vs_ma20_pct", "volume_vs_avg20", "ret_2d_pct", "ret_5d_pct", "vol_10d_pct"]


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return features (and targets) for every day. `ohlcv` has columns open, high, low, close, volume indexed by date."""
    c, o, h, l, v = (ohlcv[k] for k in ["close", "open", "high", "low", "volume"])
    out = pd.DataFrame(index=ohlcv.index)
    pc = c.shift(1)
    out["prev_close"], out["prev_open"], out["prev_high"] = pc, o.shift(1), h.shift(1)
    out["prev_low"], out["prev_volume"] = l.shift(1), v.shift(1)
    for k in (5, 10, 20):
        out[f"ma_{k}"] = pc.rolling(k).mean()                          # mean of the k closes ending at t-1
    ret = c.pct_change() * 100                                          # return of day d
    pr = ret.shift(1)                                                   # return of day t-1
    out["prev_return_pct"] = pr
    # scale-free features (comparable across price levels; used by the return-target model)
    out["prev_open_gap_pct"] = (out["prev_open"] / pc - 1) * 100       # open of t-1 vs close of t-1
    out["prev_range_pct"] = (out["prev_high"] - out["prev_low"]) / pc * 100
    for k in (5, 10, 20):
        out[f"close_vs_ma{k}_pct"] = (pc / out[f"ma_{k}"] - 1) * 100
    out["volume_vs_avg20"] = out["prev_volume"] / v.shift(1).rolling(20).mean()
    out["ret_2d_pct"] = (pc / c.shift(3) - 1) * 100                     # return over days t-3 -> t-1
    out["ret_5d_pct"] = (pc / c.shift(6) - 1) * 100
    out["vol_10d_pct"] = pr.rolling(10).std()                           # std of the last 10 daily returns (ending t-1)
    out["close"] = c
    out["target_return_pct"] = (c / pc - 1) * 100
    return out


def training_table(ohlcv: pd.DataFrame) -> pd.DataFrame:
    t = build_features(ohlcv).iloc[HISTORY_DAYS:]
    assert not t.isna().any().any()
    return t


def features_for_prediction(history: pd.DataFrame) -> pd.DataFrame:
    """Single-row features for the day AFTER the last row of `history` (exactly HISTORY_DAYS rows, oldest first).

    The next day is appended as a placeholder row and the SAME build_features() is run; only that last row is returned.
    """
    if len(history) != HISTORY_DAYS:
        raise ValueError(f"need exactly {HISTORY_DAYS} trading days of history, got {len(history)}")
    nxt = history.index[-1] + pd.offsets.BDay(1)
    frame = pd.concat([history[["open", "high", "low", "close", "volume"]].astype(float),
                       pd.DataFrame({k: [np.nan] for k in ["open", "high", "low", "close", "volume"]}, index=[nxt])])
    return build_features(frame).iloc[[-1]].drop(columns=["close", "target_return_pct"])
