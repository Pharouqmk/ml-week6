"""Build a real example request: the 20 trading days up to (and including) --asof for AAPL.
Usage: python scripts/make_example_request.py [--asof 2017-12-27] [--out docs/example_request.json]"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep import load_raw, get_stock


def build(asof: str = "2017-12-27", symbol: str = "AAPL"):
    s = get_stock(load_raw(), symbol)
    pos = s.index.get_loc(pd.Timestamp(asof))
    hist = s.iloc[pos - 19: pos + 1]
    payload = {"symbol": symbol, "history": [{"date": str(i.date()), "open": round(float(r.open), 4), "high": round(float(r.high), 4),
                                              "low": round(float(r.low), 4), "close": round(float(r.close), 4), "volume": int(r.volume)} for i, r in hist.iterrows()]}
    nxt = s.iloc[pos + 1]
    return payload, float(nxt.close), str(s.index[pos + 1].date())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--asof", default="2017-12-27"); ap.add_argument("--out", default=str(ROOT / "docs" / "example_request.json"))
    a = ap.parse_args(); payload, actual, nxt = build(a.asof)
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {a.out} | actual AAPL close on the next trading day ({nxt}): {actual:.2f}")
