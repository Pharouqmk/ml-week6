"""Train, compare and track next-trading-day closing price models with MLflow.

Run from the project root:   python src/train.py            (default stock: AAPL)
Outputs: MLflow runs (mlflow.db + mlartifacts/), models/xgb_<symbol>_next_close.json, models/model_card.json,
         data/processed/*.csv, reports/*.json|csv

Chronological split (never shuffled):
  train      : start .. 2016-06-30
  validation : 2016-07-01 .. 2016-12-31   (model selection, early stopping)
  test       : 2017-01-01 .. 2017-12-29   (used once, for the final report)

Two XGBoost families are compared (both end up predicting a PRICE):
  direct  : target = next close, features = the nine required price/volume/moving-average features
  return  : target = next-day % return, scale-free features; price = previous close x (1 + predicted return)
"""
import argparse
import json
import platform
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep import load_raw, audit_dataset, get_stock                       # noqa: E402
from src.features import training_table, REQUIRED_FEATURES, RATIO_FEATURES        # noqa: E402
from src.metrics import evaluate                                                    # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
SPLITS = {"train_end": "2016-06-30", "val_start": "2016-07-01", "val_end": "2016-12-31", "test_start": "2017-01-01"}
DIRECT_CONFIGS = [{"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
                  {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
                  {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}]
RETURN_SPACE = {"max_depth": [1, 2, 3, 4], "learning_rate": [0.01, 0.02, 0.05], "min_child_weight": [5, 10, 20, 40],
                "subsample": [0.5, 0.7, 0.9], "colsample_bytree": [0.5, 0.7, 1.0], "reg_lambda": [1.0, 5.0, 20.0], "gamma": [0.0, 0.5]}
N_SEARCH = 20


def split(table):
    return (table.loc[:SPLITS["train_end"]], table.loc[SPLITS["val_start"]:SPLITS["val_end"]], table.loc[SPLITS["test_start"]:])


def fit_xgb(params, feats, target, tr, va=None, n_estimators=None):
    p = {"n_estimators": 2000, **params}
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    model = xgb.XGBRegressor(**p, random_state=SEED, tree_method="hist", n_jobs=4,
                             early_stopping_rounds=50 if n_estimators is None else None, eval_metric="rmse")
    if n_estimators is None:
        model.fit(tr[feats], tr[target], eval_set=[(va[feats], va[target])], verbose=False)
    else:
        model.fit(tr[feats], tr[target], verbose=False)
    return model


def to_price(model_kind, raw_pred, prev_close):
    return np.asarray(raw_pred) if model_kind == "direct" else np.asarray(prev_close) * (1 + np.asarray(raw_pred) / 100)


def log_metrics(prefix, m):
    for k, v in m.items():
        if v is not None and np.isfinite(v):
            mlflow.log_metric(f"{prefix}_{k}", v)


def make_portable(db_path: Path, root: Path):
    """Store artifact paths relative to the project root so the MLflow evidence still opens after moving or zipping."""
    prefix = f"file://{root.as_posix()}/"
    con = sqlite3.connect(db_path)
    for table, col in [("experiments", "artifact_location"), ("runs", "artifact_uri"), ("logged_models", "artifact_location")]:
        con.execute(f"UPDATE {table} SET {col} = REPLACE({col}, ?, '') WHERE {col} LIKE ?", (prefix, prefix + "%"))
    con.commit(); con.close()


def run_experiments(symbol="AAPL", tracking_uri=None, write_outputs=True):
    experiment = f"{symbol.lower()}-next-day-close"
    tracking_uri = tracking_uri or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)

    raw = load_raw()
    audit = audit_dataset(raw)
    stock = get_stock(raw, symbol)
    table = training_table(stock)
    tr, va, te = split(table)
    print(f"{symbol}: {len(stock)} days | table {len(table)} | train {len(tr)} | val {len(va)} | test {len(te)}")
    results = {"symbol": symbol, "audit": audit, "splits": {"train": [str(tr.index.min().date()), str(tr.index.max().date()), len(tr)],
               "validation": [str(va.index.min().date()), str(va.index.max().date()), len(va)], "test": [str(te.index.min().date()), str(te.index.max().date()), len(te)]},
               "price_range": {"train_val": [float(pd.concat([tr, va]).close.min()), float(pd.concat([tr, va]).close.max())], "test": [float(te.close.min()), float(te.close.max())]}, "runs": {}}
    tags = lambda kind: {"symbol": symbol, "run_kind": kind, "target": "next trading day close (USD, assumed)"}

    def score(part, raw_pred, kind):
        return evaluate(part["close"], to_price(kind, raw_pred, part["prev_close"]), part["prev_close"])

    # ---- 1. baselines (nothing is fitted, so logging test metrics is safe)
    drift = float(tr.target_return_pct.mean())
    for name, fn in [("baseline_naive_last_close", lambda d: d.prev_close), ("baseline_moving_average_5", lambda d: d.ma_5),
                     ("baseline_moving_average_20", lambda d: d.ma_20), ("baseline_last_close_plus_train_drift", lambda d: d.prev_close * (1 + drift / 100))]:
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(tags("baseline")); mlflow.log_param("rule", name.replace("baseline_", "").replace("_", " "))
            mv, mt = evaluate(va.close, fn(va), va.prev_close), evaluate(te.close, fn(te), te.prev_close)
            log_metrics("val", mv); log_metrics("test", mt); results["runs"][name] = {"val": mv, "test": mt}

    # ---- 2. ridge regressions (linear models extrapolate, so they are a fair non-tree comparison)
    for name, feats, target, kind in [("ridge_direct_price", REQUIRED_FEATURES, "close", "direct"), ("ridge_return_target", RATIO_FEATURES, "target_return_pct", "return")]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[feats], tr[target])
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(tags("linear-model")); mlflow.log_params({"model": "StandardScaler+Ridge", "alpha": 10.0, "target_type": kind, "n_features": len(feats)})
            mv = score(va, m.predict(va[feats]), kind); log_metrics("val", mv)
            results["runs"][name] = {"val": mv}

    # ---- 3. XGBoost direct price (as assigned) and 4. XGBoost return target (random search)
    candidates = []
    for i, cfg in enumerate(DIRECT_CONFIGS):
        with mlflow.start_run(run_name=f"xgb_direct_price_{i}"):
            mlflow.set_tags({**tags("xgboost-direct"), "target_type": "direct"}); mlflow.log_params({**cfg, "target_type": "direct", "n_features": len(REQUIRED_FEATURES)})
            m = fit_xgb(cfg, REQUIRED_FEATURES, "close", tr, va)
            mv = score(va, m.predict(va[REQUIRED_FEATURES]), "direct"); log_metrics("val", mv); mlflow.log_metric("best_iteration", int(m.best_iteration))
            candidates.append({"run": f"xgb_direct_price_{i}", "kind": "direct", "feats": REQUIRED_FEATURES, "target": "close", "params": cfg, "val": mv, "best_iteration": int(m.best_iteration)})
    rng = np.random.default_rng(SEED)
    for i in range(N_SEARCH):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in RETURN_SPACE.items()}
        cfg = {k: (x.item() if hasattr(x, "item") else x) for k, x in cfg.items()}
        with mlflow.start_run(run_name=f"xgb_return_{i:02d}"):
            mlflow.set_tags({**tags("xgboost-return"), "target_type": "return"}); mlflow.log_params({**cfg, "target_type": "return", "n_features": len(RATIO_FEATURES)})
            m = fit_xgb(cfg, RATIO_FEATURES, "target_return_pct", tr, va)
            mv = score(va, m.predict(va[RATIO_FEATURES]), "return"); log_metrics("val", mv); mlflow.log_metric("best_iteration", int(m.best_iteration))
            candidates.append({"run": f"xgb_return_{i:02d}", "kind": "return", "feats": RATIO_FEATURES, "target": "target_return_pct", "params": cfg, "val": mv, "best_iteration": int(m.best_iteration)})
    best_direct = min([c for c in candidates if c["kind"] == "direct"], key=lambda c: c["val"]["RMSE"])
    best = min(candidates, key=lambda c: c["val"]["RMSE"])
    naive_val = results["runs"]["baseline_naive_last_close"]["val"]
    results["selection"] = {"best_run": best["run"], "kind": best["kind"], "params": best["params"], "best_iteration": best["best_iteration"], "val": best["val"],
                            "best_direct_run": best_direct["run"], "best_direct_val": best_direct["val"], "naive_val": naive_val,
                            "beats_naive_on_validation": bool(best["val"]["RMSE"] < naive_val["RMSE"])}
    print("best learned model on validation:", best["run"], round(best["val"]["RMSE"], 3), "| naive:", round(naive_val["RMSE"], 3))
    results["candidates"] = [{k: c[k] for k in ("run", "kind", "val", "best_iteration")} for c in candidates]

    # ---- 5. finals: refit on train+validation (no early stopping), score ONCE on test
    trva = pd.concat([tr, va])
    finals = {}
    for label, c in [("selected", best), ("direct_as_assigned", best_direct)]:
        if label == "direct_as_assigned" and best_direct["run"] == best["run"]:
            finals[label] = finals["selected"]; continue
        n_final = int((c["best_iteration"] + 1) * 1.1)
        model = fit_xgb(c["params"], c["feats"], c["target"], trva, n_estimators=n_final)
        raw_pred = model.predict(te[c["feats"]]); price = to_price(c["kind"], raw_pred, te["prev_close"])
        finals[label] = {"c": c, "model": model, "n_final": n_final, "price": price, "metrics": evaluate(te.close, price, te.prev_close)}
        with mlflow.start_run(run_name=f"final_{label}_train+val") as run:
            mlflow.set_tags({**tags("xgboost-final"), "target_type": c["kind"], "selected_from": c["run"], "best": "true" if label == "selected" else "false"})
            mlflow.log_params({**c["params"], "n_estimators": n_final, "target_type": c["kind"], "fit_on": "train+validation", "n_features": len(c["feats"])})
            log_metrics("test", finals[label]["metrics"])
            sig = infer_signature(te[c["feats"]].head(5), model.predict(te[c["feats"]].head(5)))
            mlflow.xgboost.log_model(model, name="model", signature=sig)
            imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5)); imp.tail(12).plot(kind="barh", ax=ax, color="#1F4E79")
            ax.set_title(f"Feature importance (gain): {label}"); ax.set_xlabel("Gain"); ax.set_ylabel("Feature")
            mlflow.log_figure(fig, "feature_importance.png"); plt.close(fig)
            finals[label]["run_id"] = run.info.run_id
    sel = finals["selected"]
    # paired test of absolute errors against the naive baseline on the test year
    err_model = np.abs(te.close.values - sel["price"]); err_naive = np.abs(te.close.values - te.prev_close.values)
    w = stats.wilcoxon(err_naive - err_model) if np.any(err_naive != err_model) else None
    results["test"] = {"selected": sel["metrics"], "direct_as_assigned": finals["direct_as_assigned"]["metrics"],
                       "naive": results["runs"]["baseline_naive_last_close"]["test"], "wilcoxon_p_vs_naive": float(w.pvalue) if w else None,
                       "share_days_model_closer_than_naive": float(np.mean(err_model < err_naive)),
                       "share_test_days_above_train_val_max": float((te.close > trva.close.max()).mean())}
    rows = [("Baseline: repeat last close", results["runs"]["baseline_naive_last_close"]["test"]),
            ("Baseline: 5-day moving average", results["runs"]["baseline_moving_average_5"]["test"]),
            ("Baseline: 20-day moving average", results["runs"]["baseline_moving_average_20"]["test"]),
            ("Baseline: last close + training drift", results["runs"]["baseline_last_close_plus_train_drift"]["test"]),
            (f"XGBoost direct price ({finals['direct_as_assigned']['c']['run']})", finals["direct_as_assigned"]["metrics"]),
            (f"XGBoost {best['kind']} target ({best['run']}) - selected", sel["metrics"])]
    summary = pd.DataFrame([{"model": n, **m} for n, m in rows]).round(4)
    results["test_summary"] = summary.to_dict("records")
    results["final_run_id"] = sel["run_id"]

    if write_outputs:
        (ROOT / "models").mkdir(exist_ok=True)
        model_path = ROOT / "models" / f"xgb_{symbol.lower()}_next_close.json"
        sel["model"].save_model(model_path)
        c = sel["c"]
        card = {"model_name": f"xgb_{symbol.lower()}_next_close", "version": "1.0.0", "symbol": symbol, "target_type": c["kind"], "features": c["feats"],
                "task": "predict the closing price of the next trading day from the previous 20 trading days (OHLCV)",
                "currency_note": "prices assumed to be US dollars; the file has no currency column",
                "params": {**c["params"], "n_estimators": sel["n_final"]}, "training_window": [str(trva.index.min().date()), str(trva.index.max().date())],
                "test_window": [str(te.index.min().date()), str(te.index.max().date())], "validation_metrics": c["val"], "test_metrics": sel["metrics"],
                "naive_baseline_test_metrics": results["runs"]["baseline_naive_last_close"]["test"],
                "beats_naive_on_test_rmse": bool(sel["metrics"]["RMSE"] < results["runs"]["baseline_naive_last_close"]["test"]["RMSE"]),
                "price_range_train_val": results["price_range"]["train_val"], "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "libraries": {"python": platform.python_version(), "xgboost": xgb.__version__, "mlflow": mlflow.__version__, "pandas": pd.__version__},
                "mlflow_run_id": sel["run_id"], "mlflow_experiment": experiment,
                "disclaimer": "Historical statistical model for education. Not investment advice; historical patterns do not guarantee future prices."}
        json.dump(card, open(ROOT / "models" / "model_card.json", "w"), indent=2)
        proc = ROOT / "data" / "processed"; proc.mkdir(parents=True, exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
        stock.to_csv(proc / f"{symbol.lower()}_clean.csv"); table.to_csv(proc / f"{symbol.lower()}_features.csv")
        pd.DataFrame({"date": te.index, "actual_close": te.close.values, "prev_close": te.prev_close.values, "xgb_selected": sel["price"],
                      "xgb_direct": finals["direct_as_assigned"]["price"], "ma_5": te.ma_5.values}).to_csv(proc / "test_predictions.csv", index=False)
        summary.to_csv(ROOT / "reports" / "test_metrics_summary.csv", index=False)
        json.dump(results, open(ROOT / "reports" / "experiment_results.json", "w"), indent=2, default=str)
        mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"]).to_csv(ROOT / "reports" / "mlflow_runs_export.csv", index=False)
    if tracking_uri == f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}":
        make_portable(ROOT / "mlflow.db", ROOT)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--tracking-uri", default=None, help="default: sqlite:///<project>/mlflow.db")
    a = ap.parse_args()
    r = run_experiments(a.symbol, a.tracking_uri)
    print(pd.DataFrame(r["test_summary"]).to_string(index=False))
