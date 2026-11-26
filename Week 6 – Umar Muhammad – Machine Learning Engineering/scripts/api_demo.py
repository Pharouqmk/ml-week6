"""Calls a RUNNING API (default http://127.0.0.1:8000) with a valid and several invalid requests and saves the transcript.
Usage: uvicorn app.main:app   (other terminal)   then   python scripts/api_demo.py"""
import copy, json, sys
from datetime import datetime
from pathlib import Path
import httpx
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_example_request import build

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
out = [f"API transcript generated {datetime.now():%Y-%m-%d %H:%M:%S} against {BASE}\n"]


def show(title, method, path, payload=None, note=None):
    r = httpx.request(method, BASE + path, json=payload, timeout=30)
    out.append(f"### {title}\n{method} {path}" + (f"\nrequest: {note}" if note else "") + f"\n-> HTTP {r.status_code}\n" + json.dumps(r.json(), indent=2)[:1500] + "\n")
    return r


payload, actual, nxt = build("2017-12-27")
show("1. Health check", "GET", "/health")
info = httpx.get(BASE + "/model-info").json()
out.append("### 2. Model info (abridged)\nGET /model-info\n-> HTTP 200\n" + json.dumps({k: info[k] for k in ("model_name", "symbol", "target_type", "training_window", "test_window", "beats_naive_on_test_rmse")}
           | {"test_metrics": {k: round(v, 3) for k, v in info["test_metrics"].items()}}, indent=2) + "\n")
r = show("3. VALID prediction (20 real AAPL trading days up to 27 Dec 2017)", "POST", "/predict", payload, "symbol=AAPL, 20 daily OHLCV records, newest = 2017-12-27")
b = r.json()
out.append(f"   actual AAPL close on {nxt}: {actual:.2f} | model: {b['predicted_next_close']:.2f} | baseline (repeat last close): {b['baseline_next_close']:.2f}\n")
bad = {"4. INVALID: wrong stock symbol (MSFT)": lambda p: p.__setitem__("symbol", "MSFT"),
       "5. INVALID: only 19 days of history": lambda p: p.__setitem__("history", p["history"][:19]),
       "6. INVALID: high price below low price": lambda p: p["history"][5].update({"high": 100.0, "low": 150.0}),
       "7. INVALID: negative volume": lambda p: p["history"][2].__setitem__("volume", -1),
       "8. INVALID: dates out of order": lambda p: p["history"].reverse()}
for title, mut in bad.items():
    p = copy.deepcopy(payload); mut(p); show(title, "POST", "/predict", p, "same request as #3 with one thing broken")
text = "\n".join(out); (ROOT / "docs" / "api_evidence.txt").write_text(text); print(text)
