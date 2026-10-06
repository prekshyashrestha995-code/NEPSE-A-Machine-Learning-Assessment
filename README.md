# Replication files: Do Foreign Stock Markets Help Forecast the Nepal Stock Exchange?

Code, data and outputs for the paper *Do Foreign Stock Markets Help Forecast the
Nepal Stock Exchange? A Machine-Learning Assessment* (Prekshya Shrestha).

The study asks whether daily movements in 27 foreign equity markets improve
day-ahead forecasts of NEPSE log returns beyond NEPSE's own trading history. Six
information sets are compared across eight models (OLS, Ridge, Random Forest,
Extra Trees, Gradient Boosting, XGBoost, LSTM, GRU) on a chronological hold-out
(test period 2 May 2023 – 4 August 2026, 731 trading days).

## Contents

```
NEPSE_replication/
├── README.md
├── requirements.txt          packages for notebooks 01, 02 and 04
├── requirements-dl.txt       adds TensorFlow for notebook 03
├── data/raw/
│   ├── NEPSE SHARESHANSER.xlsx           NEPSE OHLC and turnover (ShareSansar)
│   └── all_countries_index_levels.csv    daily index levels, 33 foreign markets + NEPSE
├── src/nepse/                shared code, imported by every notebook
│   ├── config.py             paths, sample dates, market lists, model settings
│   ├── data.py               loading, cleaning, coverage screen, lead-lag correlations
│   ├── features.py           feature blocks, PCA/PLS factors, the six information sets
│   ├── models.py             OLS, Ridge, RF, ET, GBR, XGBoost and the time-series CV search
│   ├── deep.py               LSTM / GRU
│   ├── metrics.py            RMSE, MAE, R², directional accuracy, Pesaran–Timmermann, Diebold–Mariano
│   ├── experiments.py        model fitting, walk-forward check, result storage
│   └── plots.py              Figures 1 and 3–6
├── notebooks/                run in order 01 → 04
└── results/
    ├── figures/              every figure in the paper (PNG and PDF)
    └── tables/               stored model results and every number quoted in the paper
```

## Where each part of the paper comes from

| Paper | File | Produced by |
|---|---|---|
| Section 2, coverage screen (27 of 33 markets retained) | `results/tables/coverage_screen.csv` | notebook 01 |
| Figure 1 | `results/figures/fig1_market_overview` | notebook 01 |
| Figure 2 (methodology diagram) | `results/figures/fig2_framework` | drawn from `fig2_framework.html` (not data-driven) |
| Table 1, Section 3.2 (feature counts, PCA variance) | printed by notebook 02 | `src/nepse/features.py` |
| Figure 3, Section 4.1 | `fig3_leadlag_heatmap`, `leadlag_correlations.csv`, `leadlag_ranking.csv` | notebook 01 |
| Classical model results | `metrics_classical.csv`, `predictions_classical.csv` | notebook 02 |
| LSTM / GRU results | `metrics_deep.csv`, `predictions_deep.csv` | notebook 03 |
| Walk-forward check (Sections 3.4, 4.3) | `walkforward.csv` | notebook 02 |
| Figures 4 and 5 | `fig4_rmse_heatmap`, `fig5_diracc_heatmap` | notebook 04 |
| Table 2 and Section 4.3 (Diebold–Mariano tests) | `dm_tests.csv` | notebook 04 |
| Sections 4.2, 4.4, 4.5 (all other metrics) | `all_metrics.csv`, `best_model_per_set.csv` | notebook 04 |
| Figure 6 | `fig6_pred_vs_obs_ET_Combined` | notebook 04 |
| All tables in one workbook | `paper_tables.xlsx` | notebook 04 |

## Running the code

Python 3.10 or later.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/pip install -r requirements-dl.txt     # only for notebook 03
.venv/bin/jupyter lab
```

Run the notebooks in order. Each finds the project root itself.

| Notebook | What it does | Approximate run time |
|---|---|---|
| `01_data_overview.ipynb` | coverage screen, Figure 1, Figure 3 | seconds |
| `02_model_comparison.ipynb` | fits the six classical models on all six information sets, then the walk-forward check | 10–15 minutes |
| `03_deep_learning.ipynb` | LSTM and GRU on all six information sets | a few minutes |
| `04_results_and_figures.ipynb` | Figures 4–6 and all tables, from the stored results; refits nothing | seconds |

Notebook 04 reads the stored results in `results/tables/`, so the paper's figures
and tables can be regenerated without refitting any model.

## Model settings

The defaults in `src/nepse/config.py` (`EXTENDED_SEARCH = False`) are the settings
reported in the paper:

* imputation (training-period medians) and, for OLS, Ridge and the neural networks,
  standardization are fitted on the training period only;
* Ridge: penalty chosen from {1, 10} by expanding-window time-series CV within the
  training period (α = 10 is selected for every information set);
* Random Forest and Extra Trees: 200 trees, maximum depth 6;
* Gradient Boosting: 200 trees, learning rate 0.05, maximum depth 3;
* XGBoost: 300 trees, learning rate 0.05, maximum depth 3, subsample 0.9;
* LSTM / GRU: one layer of 32 units, dropout 0.2, Adam (learning rate 0.001),
  batch size 64, 20-day input sequences, 8 epochs, last 15% of training data for validation;
* walk-forward check: gradient boosting re-estimated at 6 expanding origins.

Setting `EXTENDED_SEARCH = True` runs a larger grid search and longer neural-network
training. That configuration was not used in the paper and gives different numbers.

## Reproducibility notes

* **OLS, Ridge, Random Forest, Extra Trees and Gradient Boosting** reproduce the
  stored results exactly. This was checked by re-running notebook 02 with Python 3.13,
  NumPy 2.5, pandas 3.0 and scikit-learn 1.9. The walk-forward check uses gradient
  boosting only, so it also reproduces exactly.
* **XGBoost** predictions change slightly across XGBoost versions and platforms.
  Re-running notebook 02 can therefore give XGBoost numbers that differ a little from
  the stored ones. The stored XGBoost results are the ones reported in the paper.
* **LSTM and GRU** training is not bit-for-bit reproducible across hardware and
  TensorFlow versions. Re-running notebook 03 gives similar but not identical numbers.
* All random seeds are fixed (`config.RANDOM_STATE = 42`).

## Data

* `NEPSE SHARESHANSER.xlsx`: daily NEPSE open, high, low, close, change and
  turnover, 3 January 2010 – 4 August 2026, from ShareSansar (https://www.sharesansar.com).
* `all_countries_index_levels.csv`: daily closing levels of 33 foreign equity
  indices plus NEPSE (`Nepal` column, not used as a predictor), 3 January 2010 –
  5 August 2026, from Yahoo Finance. Six indices (Croatia, Pakistan, the
  Philippines, Poland, Romania, Sri Lanka) fail the coverage screen and are excluded.

Missing values reflect different trading calendars and gaps in coverage. They are
not zero returns, so returns are computed on genuine observation days only (see
`src/nepse/data.py`).
