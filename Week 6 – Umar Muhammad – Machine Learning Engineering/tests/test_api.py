import copy
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app
from scripts.make_example_request import build

PAYLOAD, ACTUAL, NEXT_DAY = build("2017-12-27")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    b = client.get("/model-info").json(); assert b["symbol"] == "AAPL" and "RMSE" in b["test_metrics"] and "disclaimer" in b


def test_valid_prediction_structure_and_sanity(client):
    r = client.post("/predict", json=PAYLOAD); b = r.json()
    assert r.status_code == 200
    assert b["symbol"] == "AAPL" and b["as_of_date"] == "2017-12-27"
    assert b["baseline_next_close"] == round(PAYLOAD["history"][-1]["close"], 2)
    assert abs(b["predicted_change_pct"]) < 5                       # a one-day move should be small
    assert abs(b["predicted_next_close"] - ACTUAL) / ACTUAL < 0.05
    assert "disclaimer" in b and len(b["warnings"]) >= 1


def test_lowercase_symbol_is_accepted(client):
    p = copy.deepcopy(PAYLOAD); p["symbol"] = "aapl"
    assert client.post("/predict", json=p).status_code == 200


def _bad(mut):
    p = copy.deepcopy(PAYLOAD); mut(p); return p


@pytest.mark.parametrize("name,mut", [
    ("wrong symbol", lambda p: p.__setitem__("symbol", "MSFT")),
    ("19 days only", lambda p: p.__setitem__("history", p["history"][:19])),
    ("21 days", lambda p: p.__setitem__("history", p["history"] + [p["history"][-1]])),
    ("negative price", lambda p: p["history"][3].__setitem__("close", -5)),
    ("zero price", lambda p: p["history"][3].__setitem__("open", 0)),
    ("high below low", lambda p: p["history"][5].update({"high": 100.0, "low": 150.0})),
    ("close above high", lambda p: p["history"][5].__setitem__("close", p["history"][5]["high"] + 20)),
    ("negative volume", lambda p: p["history"][2].__setitem__("volume", -1)),
    ("zero volume", lambda p: p["history"][2].__setitem__("volume", 0)),
    ("text price", lambda p: p["history"][2].__setitem__("close", "abc")),
    ("dates out of order", lambda p: p["history"].reverse()),
    ("duplicate date", lambda p: p["history"][4].__setitem__("date", p["history"][3]["date"])),
    ("bad date", lambda p: p["history"][1].__setitem__("date", "not-a-date")),
    ("missing field", lambda p: p["history"][1].pop("volume")),
    ("missing history", lambda p: p.pop("history")),
    ("missing symbol", lambda p: p.pop("symbol")),
])
def test_invalid_input_returns_422_and_no_prediction(client, name, mut):
    r = client.post("/predict", json=_bad(mut)); b = r.json()
    assert r.status_code == 422, name
    assert b["error"].startswith("Invalid input") and len(b["details"]) >= 1 and "predicted_next_close" not in b


def test_extreme_price_range_adds_warning(client):
    p = _bad(lambda p: [h.update({k: h[k] * 4 for k in ("open", "high", "low", "close")}) for h in p["history"]])
    r = client.post("/predict", json=p); assert r.status_code == 200 and any("outside the range" in w for w in r.json()["warnings"])


def test_large_daily_move_adds_warning(client):
    def mut(p):
        for h in p["history"][10:]:
            for k in ("open", "high", "low", "close"): h[k] = round(h[k] * 0.5, 4)
    r = client.post("/predict", json=_bad(mut)); assert r.status_code == 200 and any("25%" in w for w in r.json()["warnings"])
