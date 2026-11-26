# Stock Price Prediction and Model Deployment (Week 6, Machine Learning Engineering track)

**AnalystLab Africa · Experience Lab Internship · Week 6**
**Author:** Umar Muhammad · **Track:** Machine Learning Engineering (XGBoost, MLflow, FastAPI)

> **Historical data, not investment advice.** The data cover 2014 - 2017 and must not be read as current market information. Historical performance and model outputs are not guarantees of future prices.

Predicts the **closing price of the next trading day for AAPL** from the previous trading days' prices and volume, compares XGBoost with simple baselines, tracks every experiment in MLflow, and serves the selected model through a validated FastAPI endpoint.

## The main result (an honest one)
| Model (test year 2017) | MAE (USD) | RMSE (USD) | R2 |
|---|---|---|---|
| **Baseline: repeat last close** | **1.186** | **1.752** | **0.9856** |
| Baseline: 5-day moving average | 1.933 | 2.562 | 0.969 |
| Baseline: last close + training drift | 1.183 | 1.745 | 0.986 |
| XGBoost, price as target (as assigned) | 22.279 | 25.775 | -2.116 |
| XGBoost, return as target (**selected**) | 1.178 | 1.754 | 0.9856 |

- **The selected model does not beat "repeat the last close".** MAE differs by 0.7%, RMSE by 0.1%; the paired test of errors gives p = 0.17; direction accuracy (53.8%) is not better than a coin flip (p = 0.26) or than always guessing "up" (54.6%).
- **R2 of 0.986 is misleading.** Prices trend, so any method that stays near the previous price scores about 0.99. The model is therefore also scored on returns (R2 = -0.02: no skill).
- **Predicting the price directly fails** (MAE 22 USD): 89% of 2017 closes are above the highest training price, and tree models cannot predict values they have never seen. Predicting the return and converting it to a price avoids this.
- **Same story elsewhere:** walk-forward through 2017 (beats the baseline in only 5 of 12 months) and five other stocks (within about 1% of the baseline).

## Data findings
All of the assignment's data-quality statements were verified independently (497,472 rows, 505 symbols, 1,007 dates; 11/8/8 missing open/high/low; 4 zero-volume rows; 26 incomplete histories; no duplicate symbol-date pairs). Further findings:
- The 4 zero-volume rows are placeholders (open, high and low all missing). Missing prices are never replaced by zero.
- 12 rows have inconsistent prices; 39 daily moves exceed 25% and 10 exceed 40%.
- **Prices are not consistently adjusted for corporate actions:** several symbols show single-day drops of 44 - 63% that look like unadjusted splits or spin-offs, and one (LNT) has a one-day glitch. AAPL appears split-adjusted (no break at its June 2014 7-for-1 split). Dividend adjustment cannot be verified.
- The file has no currency column; prices are **assumed to be US dollars**.
- The file is ordered by date then symbol, so it is sorted by symbol and date before any calculation.

## Method
1. **Stock choice:** AAPL (complete 1,007-day history, no missing values, no corporate-action break).
2. **Features** (all from information up to the close of the previous day): previous close/open/high/low/volume, 5/10/20-day moving averages, previous return, plus scale-free extras for the return-target model. Three automated tests prove there is no future leakage and no train/serve skew.
3. **Chronological split:** train to 30 Jun 2016, validation Jul - Dec 2016, test 2017 (used once). No shuffling.
4. **Experiments (31 MLflow runs):** 4 baselines, 2 ridge models, 3 direct-price XGBoost runs, 20 return-target XGBoost runs, 2 final refits.
5. **Selection rule:** lowest validation RMSE among XGBoost runs; test year not used for selection.
6. **Evaluation:** MAE, RMSE, R2 (plus MAPE, return-level R2, direction accuracy), baseline comparison, overfitting check, walk-forward test, other stocks.
7. **API:** FastAPI with Pydantic validation; invalid input returns HTTP 422 and no prediction.

## Project structure
```
Week 6 – Umar Muhammad – Machine Learning Engineering/
├── README.md
├── requirements.txt
├── src/               data_prep.py, features.py, metrics.py, train.py (MLflow experiments)
├── app/main.py        FastAPI application
├── tests/             test_features.py (audit, leakage, consistency), test_api.py (valid/invalid requests)
├── scripts/           make_example_request.py, api_demo.py
├── notebooks/Stock_Price_Prediction_Pipeline.ipynb   full analysis, 7 figure groups
├── data/raw/          the original CSV (unchanged) and the assignment PDF is in docs/
├── data/processed/    clean AAPL series, feature table, test predictions
├── models/            xgb_aapl_next_close.json (trained model), model_card.json
├── mlflow.db, mlartifacts/                            MLflow tracking evidence
├── reports/           Technical_Report (.docx and .pdf), metric and experiment exports
├── docs/              evidence/ (MLflow table, API transcript, test output), api_evidence.txt, example_request.json, PRESENTATION_SCRIPT.md
└── images/            all figures
```

## How to run
```bash
pip install -r requirements.txt

python src/train.py                                     # runs all experiments (about 20 s), writes models/ and MLflow runs
mlflow ui --backend-store-uri sqlite:///mlflow.db       # browse experiments at http://127.0.0.1:5000 (run from the project root)

uvicorn app.main:app                                    # start the API; interactive docs at http://127.0.0.1:8000/docs
python scripts/api_demo.py                              # sends a valid and several invalid requests (second terminal)

python -m pytest tests                                  # 29 tests
```
The trained model and MLflow store are included, so the API and tests run without re-training. Re-running `src/train.py` reproduces the metrics.

## API example
`POST /predict` takes the symbol and the last **20 trading days** of open, high, low, close and volume (a real example is in `docs/example_request.json`). The service builds every feature itself with the same code used in training. The response contains the predicted close, the predicted % change, the baseline (last close), the model version, warnings and a disclaimer. Example for 27 Dec 2017: predicted 170.81, baseline 170.60, actual next close 171.08 (one day proves nothing about accuracy). Invalid input (wrong symbol, 19 days, high below low, negative volume, dates out of order, ...) returns HTTP 422 with `{"error": "Invalid input - no prediction was made", "details": [...]}`.

## Evidence checklist (Task C deliverables)
| Deliverable | Where |
|---|---|
| Data preparation and feature engineering code | `src/data_prep.py`, `src/features.py` |
| Trained XGBoost model and saved artefacts | `models/`, also logged in MLflow (`mlartifacts/`) |
| Baseline model and comparison | `reports/test_metrics_summary.csv`, notebook Sections 7 - 8 |
| Model evaluation report | `reports/Technical_Report.pdf` / `.docx` |
| MLflow experiment tracking evidence | `mlflow.db`, `mlartifacts/`, `docs/evidence/mlflow_runs_table.png`, `docs/evidence/mlflow_best_run.txt`, `reports/mlflow_runs_export.csv` |
| FastAPI application code | `app/main.py` |
| API testing evidence | `docs/api_evidence.txt`, `docs/evidence/api_transcript.png`, `docs/evidence/pytest_output.txt` |
| Technical report | `reports/Technical_Report.pdf` / `.docx` |
| README | this file |

## Limitations
Historical prices carry almost no information about tomorrow's return; prices are not verifiably adjusted for corporate actions; the currency is assumed; one stock and one test year (2017, a strong and calm year for equities); no transaction costs or risk management, so this is **not a trading model**; the API is a prototype for one stock, without authentication, monitoring or retraining.

## Integration with other tracks
This project is self-contained. It did not use or consume the outputs of any other track, and no integration is claimed.

## Work plan
Planned backwards from the Friday deadline: (1) data audit against the assignment's stated issues, (2) leakage-proof features and tests, (3) baselines and tracked experiments, (4) honest evaluation and robustness checks, (5) API and tests, (6) report, evidence, presentation and posts. Fixing the feature code and the leakage tests first meant the API could reuse the training code with no rework.

#AnalystLabAfrica
