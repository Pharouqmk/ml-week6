"""FastAPI service for the next-trading-day closing price model (AAPL).

Run from the project root:   uvicorn app.main:app --reload
Interactive docs:            http://127.0.0.1:8000/docs

The model is a historical statistical model for education. It is NOT investment advice: in testing it did not meaningfully
improve on simply repeating the last closing price, and historical patterns do not guarantee future prices.
"""
import json
import sys
from contextlib import asynccontextmanager
from datetime import date as Date
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import features_for_prediction, HISTORY_DAYS          # noqa: E402

MODEL_DIR = ROOT / "models"
CARD_PATH = MODEL_DIR / "model_card.json"
state: dict = {}


class DayRecord(BaseModel):
    date: Date
    open: float = Field(..., gt=0, le=100000)
    high: float = Field(..., gt=0, le=100000)
    low: float = Field(..., gt=0, le=100000)
    close: float = Field(..., gt=0, le=100000)
    volume: int = Field(..., gt=0, le=100_000_000_000, description="Shares traded (not a money value)")

    @model_validator(mode="after")
    def prices_are_consistent(self):
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError(f"inconsistent prices on {self.date}: high must be the highest and low the lowest of open/high/low/close")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"symbol": "AAPL", "history": [
        {"date": f"2017-12-{d:02d}", "open": 170.0 + i * 0.2, "high": 171.0 + i * 0.2, "low": 169.0 + i * 0.2, "close": 170.5 + i * 0.2, "volume": 25_000_000}
        for i, d in enumerate([1, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 26, 27, 28, 29])]}})

    symbol: str = Field(..., description="Ticker; the deployed model was trained for one stock only")
    history: list[DayRecord] = Field(..., min_length=HISTORY_DAYS, max_length=HISTORY_DAYS,
                                     description=f"The last {HISTORY_DAYS} trading days, oldest first; the newest row is the day before the prediction")

    @field_validator("symbol")
    @classmethod
    def symbol_matches_model(cls, v: str):
        v = v.strip().upper()
        if "card" in state and v != state["card"]["symbol"]:
            raise ValueError(f"this model was trained for {state['card']['symbol']} only, not {v}")
        return v

    @field_validator("history")
    @classmethod
    def dates_increase(cls, v: list[DayRecord]):
        if any(b.date <= a.date for a, b in zip(v, v[1:])):
            raise ValueError("history dates must be strictly increasing (oldest first)")
        return v


class PredictionResponse(BaseModel):
    symbol: str
    as_of_date: Date
    predicted_next_close: float
    predicted_change_pct: float
    baseline_next_close: float
    baseline_description: str = "repeat the last closing price"
    currency: str = "USD (assumed; the dataset has no currency column)"
    model_name: str
    model_version: str
    target_type: str
    warnings: list[str] = []
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CARD_PATH.exists():
        card = json.loads(CARD_PATH.read_text())
        model = xgb.XGBRegressor(); model.load_model(MODEL_DIR / f"{card['model_name']}.json")
        state["model"], state["card"] = model, card
    yield
    state.clear()


app = FastAPI(title="Next-trading-day closing price API (AAPL)", version="1.0.0", lifespan=lifespan,
              description="Educational prototype for the AnalystLab Africa Week 6 project. Historical model, not investment advice. "
                          "In testing it did not meaningfully beat repeating the last close.")


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"].replace("Value error, ", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Invalid input - no prediction was made", "details": details})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "model not loaded", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    if "card" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    c = state["card"]
    return {k: c[k] for k in ("model_name", "version", "symbol", "target_type", "task", "features", "training_window", "test_window", "test_metrics",
                              "naive_baseline_test_metrics", "beats_naive_on_test_rmse", "price_range_train_val", "currency_note", "disclaimer", "created_utc")}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    card = state["card"]
    hist = pd.DataFrame([r.model_dump() for r in req.history]); hist["date"] = pd.to_datetime(hist["date"]); hist = hist.set_index("date")
    feats = features_for_prediction(hist)
    raw = float(state["model"].predict(feats[card["features"]])[0])
    last_close = float(hist["close"].iloc[-1])
    price = raw if card["target_type"] == "direct" else last_close * (1 + raw / 100)
    lo, hi = card["price_range_train_val"]
    notes = []
    if hist["close"].min() < 0.6 * lo or hist["close"].max() > 1.4 * hi:
        notes.append("Recent prices are far outside the range the model was trained on; treat the prediction with extra caution.")
    if (hist["close"].pct_change().abs() > 0.25).any():
        notes.append("The history contains a daily move above 25%, which can indicate a split, spin-off or data error.")
    if not card["beats_naive_on_test_rmse"]:
        notes.append("In testing, this model did not meaningfully improve on repeating the last close.")
    return PredictionResponse(symbol=req.symbol, as_of_date=hist.index[-1].date(), predicted_next_close=round(price, 2),
                              predicted_change_pct=round((price / last_close - 1) * 100, 3), baseline_next_close=round(last_close, 2),
                              model_name=card["model_name"], model_version=card["version"], target_type=card["target_type"], warnings=notes,
                              disclaimer=card["disclaimer"])
