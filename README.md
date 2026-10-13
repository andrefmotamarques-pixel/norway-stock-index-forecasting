# Forecasting the Norwegian Stock Index: SARIMAX vs VAR vs LSTM

Monthly forecasting of Norway's share price index using its macro-financial links to Brent crude oil and the USD/NOK exchange rate (1992 to 2026). Norway is a major oil exporter, so the question is how much oil and the currency help predict the stock market, and whether a neural network beats classical econometric models.

Team project for the *Time Series Analysis and Forecasting* course, MSc in Data Science, ISCTE (2025/26), developed with Afonso Claudino.

## Method

- **Data:** three monthly series from FRED (share index, USD/NOK, Brent oil), 411 observations, downloaded with `src/download_data.py`.
- **Exploration and stationarity:** log transform, ADF/KPSS unit root tests, first differences. A Johansen test (see `report/main.tex`) found no cointegration, so the VAR is estimated in differences.
- **Models:**
  - SARIMAX on the log index with Brent oil as an exogenous variable (order chosen by `pmdarima` auto-ARIMA)
  - VAR(2) on the three differenced log series
  - LSTM in PyTorch on the same multivariate input
- **Evaluation:** 90/10 train/test split, out-of-sample RMSE, MAE and MAPE, plus a forward forecast.

## Results (out-of-sample)

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| SARIMAX | 65.51 | 53.90 | 20.35% |
| VAR(2) | 13.00 | 7.89 | 2.94% |
| LSTM | **11.29** | **6.87** | **2.62%** |

The multivariate models clearly beat the univariate one. The LSTM was the most accurate, with the VAR close behind and much easier to interpret.

![Out-of-sample forecast comparison](report/plots/out_of_sample_comparison.png)

## Repository structure

```
data/norway_data.csv            # merged monthly dataset
src/download_data.py            # downloads the series from FRED
src/eda_and_stationarity.py     # EDA, unit root and cointegration tests
src/models_and_evaluation.py    # SARIMAX, VAR, LSTM, evaluation and forecast
analysis.ipynb                  # notebook version with outputs
report/report_en.pdf            # full written report (English)
report/report_pt.pdf            # original report (Portuguese)
report/                         # LaTeX sources (main.tex in English, main_pt.tex in Portuguese), plots and result tables
requirements.txt                # Python dependencies
```

## How to run

```bash
pip install -r requirements.txt
python src/download_data.py          # optional, data is already included
python src/eda_and_stationarity.py
python src/models_and_evaluation.py
```

Paths are resolved from the repository root, so the scripts can be run from any folder.

## Tech stack

Python · pandas · statsmodels · pmdarima · PyTorch · scikit-learn · matplotlib · LaTeX

## Team

André Marques and Afonso Claudino. The full report with the team's methodology and discussion is in [`report/report_en.pdf`](report/report_en.pdf) (English) and [`report/report_pt.pdf`](report/report_pt.pdf) (original, Portuguese).
