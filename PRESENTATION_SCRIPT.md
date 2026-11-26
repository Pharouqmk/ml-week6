# Individual presentation script (about 7 minutes)

Talking points with approximate timings; use your own words and record your screen while showing the notebook and report.

**1. Introduction (0:30)**
Hello, I am Umar Muhammad, Machine Learning Engineering track. My task: predict the next trading day's closing price for one stock, AAPL, using S&P 500 data from 2014 to 2017, track experiments with MLflow, and deploy the model with FastAPI. This is historical data and a learning project, not investment advice.

**2. Data audit (1:15)**
- I verified every data-quality statement in the brief myself: 497,472 rows, 505 symbols, 1,007 dates, 11/8/8 missing open/high/low, 4 zero-volume rows, 26 incomplete histories, no duplicates.
- The zero-volume rows are placeholders with all prices missing. I never replace missing prices with zero.
- Prices are not consistently adjusted: some symbols show single-day drops of 44 to 63% that look like unadjusted splits or spin-offs. AAPL appears split-adjusted. Currency is assumed to be US dollars.
- I chose AAPL because its history is complete and clean.

**3. Features and leakage (1:00)**
- Every feature uses data up to the previous day only; moving averages are shifted by one day.
- Three automated tests prove no future information enters the features and that the API builds exactly the same features as training.

**4. Split and baseline (1:00)**
- Chronological split: train to mid-2016, validation to end of 2016, test is 2017, used once.
- The key baseline is "repeat the last close". And look at this chart: in 2017 AAPL closes above every training price on 89% of days (show Figure 3).

**5. Results (1:45)**
- The selected XGBoost, which predicts the return and converts it to a price, matches the baseline: MAE 1.18 vs 1.19 dollars. R-squared is 0.986 for both, so R-squared on prices is misleading.
- The model has no skill on returns (R-squared about zero), is closer than the baseline on only 53% of days, and its direction accuracy of 53.8% is no better than a coin flip.
- The XGBoost that predicts the price directly fails badly: 22 dollar error, because trees cannot predict prices above those they have seen (show Figure 5).
- Walk-forward testing and five other stocks give the same conclusion.

**6. Deployment (0:45)**
- FastAPI service with health, model-info and predict endpoints; it validates the symbol, exactly 20 days, positive prices, consistent high and low, and ordered dates; bad input returns a clear 422 with no prediction.
- 29 automated tests pass; responses include the baseline, warnings and a disclaimer.

**7. Limitations, recommendations and contribution (0:45)**
- Historical prices hold almost no information about tomorrow's return; not a trading model.
- Recommendations: always compare against the naive baseline and score returns, predict changes not price levels with tree models, use adjusted prices, and try volatility forecasting instead.
- My contribution: the whole pipeline, tests, API and documentation. The project is self-contained; I do not claim integration with other tracks.
