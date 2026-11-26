"""Loading, auditing and selecting stock data from the S&P 500 2014-2017 file.

The original CSV is never modified. Currency: the file has no currency column; the symbols are US-listed stocks, so prices
are ASSUMED to be US dollars (to be confirmed from the source documentation). Corporate actions: no adjusted-close field is
supplied; evidence in the data (see audit) shows prices are NOT consistently adjusted, so this is treated as a limitation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "S_P_500_Stock_Prices_2014-2017.csv"
PRICE_COLS = ["open", "high", "low", "close"]


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the file, parse dates (month/day/year) and sort by symbol then date (the file is not in that order)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def audit_dataset(df: pd.DataFrame) -> dict:
    """Independent checks of every data-quality statement in the assignment."""
    counts = df.groupby("symbol").size()
    ret = df.groupby("symbol")["close"].pct_change() * 100
    ohlc_bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    zero_vol = df[df["volume"] == 0]
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]), "symbols": int(df["symbol"].nunique()),
        "dates": int(df["date"].nunique()), "start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
        "missing": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
        "duplicate_symbol_date": int(df.duplicated(["symbol", "date"]).sum()),
        "zero_volume_rows": int(len(zero_vol)), "zero_volume_all_prices_missing": bool(zero_vol[["open", "high", "low"]].isna().all().all()) if len(zero_vol) else None,
        "symbols_full_history": int((counts == df["date"].nunique()).sum()), "symbols_incomplete": int((counts < df["date"].nunique()).sum()),
        "shortest_history": {"symbol": str(counts.idxmin()), "days": int(counts.min())},
        "ohlc_inconsistent_rows": int(ohlc_bad.sum()),
        "moves_over_25pct": int((ret.abs() > 25).sum()), "moves_over_40pct": int((ret.abs() > 40).sum()),
        "nonpositive_prices": int((df[PRICE_COLS] <= 0).sum().sum()),
    }


def get_stock(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One stock, indexed by date, strictly chronological, with complete OHLCV (asserted)."""
    s = df[df["symbol"] == symbol].set_index("date").sort_index().drop(columns="symbol")
    assert len(s) > 0, f"unknown symbol {symbol}"
    assert s.index.is_unique and s.index.is_monotonic_increasing
    assert not s.isna().any().any(), f"{symbol} has missing values; choose a symbol with a clean history"
    assert (s[PRICE_COLS] > 0).all().all() and (s["volume"] > 0).all()
    return s
