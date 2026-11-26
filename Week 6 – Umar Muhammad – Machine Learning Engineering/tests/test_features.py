import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep import load_raw, audit_dataset, get_stock
from src.features import build_features, features_for_prediction, training_table

RAW = load_raw()
STOCK = get_stock(RAW, "AAPL").astype(float)
DROP = ["close", "target_return_pct"]


def test_audit_reproduces_the_assignments_data_quality_statements():
    a = audit_dataset(RAW)
    assert a["rows"] == 497472 and a["symbols"] == 505 and a["dates"] == 1007
    assert a["missing"] == {"open": 11, "high": 8, "low": 8} and a["zero_volume_rows"] == 4 and a["duplicate_symbol_date"] == 0
    assert a["symbols_incomplete"] == 26


def test_stock_is_chronological_and_complete():
    assert STOCK.index.is_monotonic_increasing and STOCK.index.is_unique and len(STOCK) == 1007 and not STOCK.isna().any().any()


def test_features_do_not_use_future_information():
    k = 500
    altered = STOCK.copy(); altered.iloc[k:] = altered.iloc[k:] * 1.7
    a = build_features(STOCK).iloc[:k].drop(columns=DROP).dropna(); b = build_features(altered).iloc[:k].drop(columns=DROP).dropna()
    assert np.allclose(a.values, b.values)


def test_day_t_does_not_influence_its_own_features():
    h = 400
    altered = STOCK.copy(); altered.iloc[h] = altered.iloc[h] * 3
    assert np.allclose(build_features(STOCK).iloc[h].drop(DROP).values.astype(float), build_features(altered).iloc[h].drop(DROP).values.astype(float))


def test_previous_day_features_are_really_the_previous_day():
    f = build_features(STOCK)
    assert np.isclose(f["prev_close"].iloc[10], STOCK["close"].iloc[9]) and np.isclose(f["prev_volume"].iloc[10], STOCK["volume"].iloc[9])
    assert np.isclose(f["ma_5"].iloc[10], STOCK["close"].iloc[5:10].mean())


def test_api_features_match_training_features():
    i = 300
    fp = features_for_prediction(STOCK.iloc[i - 20:i]); ft = build_features(STOCK).iloc[[i]].drop(columns=DROP)
    assert np.allclose(fp.values.astype(float), ft.values.astype(float))


def test_training_table_has_no_missing_values():
    assert not training_table(STOCK).isna().any().any()
