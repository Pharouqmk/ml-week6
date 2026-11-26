import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(actual, predicted, prev_close=None) -> dict:
    """Price metrics plus, when prev_close is given, return-level skill and direction accuracy."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    m = {"MAE": float(mean_absolute_error(a, p)), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
         "R2": float(r2_score(a, p)), "MAPE_pct": float(np.mean(np.abs((a - p) / a)) * 100)}
    if prev_close is not None:
        pc = np.asarray(prev_close, float)
        ra, rp = (a / pc - 1) * 100, (p / pc - 1) * 100
        m["R2_of_returns"] = float(r2_score(ra, rp))
        if not np.allclose(p, pc):                      # "repeat last close" predicts no direction at all
            m["direction_accuracy_pct"] = float(np.mean(((a - pc) > 0) == ((p - pc) > 0)) * 100)
    return m
