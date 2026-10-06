"""Paths, sample definition, market lists and run settings.

The defaults reproduce the results reported in the paper. A notebook may
override a setting before running, e.g.

    from nepse import config
    config.EXTENDED_SEARCH = True     # larger search, NOT the paper's setting
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths.  This file is <project>/src/nepse/config.py, so the root is two up.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

PANEL_CSV = DATA_DIR / "all_countries_index_levels.csv"
NEPSE_XLSX = DATA_DIR / "NEPSE SHARESHANSER.xlsx"


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for d in (RESULTS_DIR, FIGURE_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------
# EXTENDED_SEARCH = False reproduces the paper:
#   * OLS: no hyperparameters; Ridge: alpha chosen from {1, 10} by 2-fold
#     expanding-window CV inside the training period;
#   * RF / ET: 200 trees, max depth 6; GBR: 200 trees, learning rate 0.05,
#     depth 3; XGBoost: 300 trees, learning rate 0.05, depth 3, subsample 0.9;
#   * LSTM / GRU: 8 training epochs.
# True runs a larger 3-fold CV grid search and up to 120 epochs with early
# stopping. It was not used for the paper and gives different numbers.
EXTENDED_SEARCH = False
FIX_ROLLING = True    # rolling features on each series' own trading days
RANDOM_STATE = 42

import pandas as _pd  # noqa: E402  (only needed for the Timestamps below)

TRAIN_START = _pd.Timestamp("2010-01-01")
TEST_START = _pd.Timestamp("2023-05-01")
SAMPLE_END = _pd.Timestamp("2026-08-31")   # sample cap; data actually ends 2026-08

RET_SCALE = 1e4       # log returns -> basis points

# Information set and model shown in the paper's predicted-vs-observed figure.
DIAG_SET = "Combined"
DIAG_MODEL = "ET"

# Walk-forward robustness check (Section 3.4 of the paper)
WALKFORWARD_MODEL = 'GBR'
WALKFORWARD_ORIGINS = 6

# Deep learning
LOOKBACK = 20
DL_CONFIG = dict(units=32, dropout=0.2, lr=1e-3, batch=64,
                 max_epochs=8, patience=12, val_frac=0.15)
DL_MAX_EPOCHS_EXTENDED = 120   # replaces DL_CONFIG['max_epochs'] when EXTENDED_SEARCH=True

# --------------------------------------------------------------------------
# Markets
# --------------------------------------------------------------------------
# 27 foreign markets retained after the coverage screen (dropped: Philippines,
# Pakistan, SriLanka, Romania, Croatia, Poland - stale or truncated series).
SELECTED = ['Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany', 'Greece',
            'HongKong', 'India', 'Indonesia', 'Italy', 'Japan', 'Malaysia', 'Mexico',
            'Netherlands', 'SaudiArabia', 'Singapore', 'SouthAfrica', 'SouthKorea',
            'Spain', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'UK', 'USA']

REGION = {
    'NorthAmerica': ['USA', 'Canada', 'Mexico'],
    'LatAm': ['Brazil'],
    'WesternEurope': ['France', 'Germany', 'Netherlands', 'Switzerland', 'UK'],
    'SouthernEurope': ['Italy', 'Spain', 'Greece'],
    'NorthernEurope': ['Sweden'],
    'EastAsia': ['China', 'HongKong', 'Japan', 'SouthKorea', 'Taiwan'],
    'SEAsia': ['Singapore', 'Malaysia', 'Thailand', 'Indonesia'],
    'SouthAsiaME': ['India', 'SaudiArabia', 'Turkey'],
    'Africa': ['SouthAfrica'],
}

# NEPSE is column 'Nepal' in the panel file.
NEPSE_PANEL_COL = 'Nepal'

# --------------------------------------------------------------------------
# Information sets: internal key -> label used in every table and figure
# --------------------------------------------------------------------------
INFO_LABELS = {
    'NEPSE-only':     'NEPSE only',
    'NEPSE+PCA':      'NEPSE + global factors',
    'NEPSE+PLS':      'NEPSE + global factors (PLS)',
    'GlobalPCA-only': 'Global factors only',
    'Foreign-only':   'Foreign only',
    'Combined':       'Combined',
}
INFO_ORDER = list(INFO_LABELS)

# Subset shown in the paper's heatmaps, Figures 4-5 (PLS left out for readability).
PAPER_INFO_ORDER = ['NEPSE-only', 'NEPSE+PCA', 'GlobalPCA-only', 'Foreign-only', 'Combined']

# Row order for the results matrices: simple to complex, classical then deep.
MODEL_ORDER = ['OLS', 'Ridge', 'RF', 'ET', 'GBR', 'XGB', 'LSTM', 'GRU']

# --------------------------------------------------------------------------
# Figure colours
# --------------------------------------------------------------------------
OBS_COLOR = "#1f3b4d"
PRED_COLOR = "#e07b39"
ACCENT_COLOR = "#3b7dd8"
NEPSE_COLOR = "#b2182b"
PEER_COLOR = "#4d4d4d"
