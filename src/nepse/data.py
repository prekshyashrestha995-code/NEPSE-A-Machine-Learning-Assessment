"""Loading and cleaning of the two raw inputs.

* ``all_countries_index_levels.csv`` - daily index levels, 34 markets.
* ``NEPSE SHARESHANSER.xlsx``        - NEPSE OHLC + turnover from ShareSansar.

The cleaning rules are the ones used throughout the study:

1. isolated one-day spikes in an index level are replaced by the geometric
   mean of their neighbours;
2. returns are computed only on *genuine* observation days - a level that is
   identical to the previous one is treated as a non-trading / carried-forward
   day and produces no return, rather than a spurious zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def remove_isolated_spikes(series, spike_ratio=2.0, neighbor_tolerance=0.30):
    """Replace isolated erroneous observations with the neighbours' geometric mean.

    A point is flagged when it sits far from both neighbours while the two
    neighbours themselves agree with each other.
    """
    s = pd.to_numeric(series, errors='coerce').copy()
    s[s <= 0] = np.nan                                  # log scale needs positives

    log_s = np.log(s)
    previous, following = log_s.shift(1), log_s.shift(-1)
    expected = (previous + following) / 2
    isolated = ((abs(log_s - expected) > np.log(spike_ratio)) &
                (abs(previous - following) < np.log(1 + neighbor_tolerance)))
    s.loc[isolated] = np.exp(expected.loc[isolated])
    return s


def genuine_level(level_series):
    """Cleaned index level with carried-forward (unchanged) values set to NaN."""
    s = remove_isolated_spikes(level_series).where(lambda x: x > 0)
    return s.where(s.diff() != 0)


def genuine_returns(level_series):
    """Daily log returns on genuine observation days only, on the original index."""
    s = remove_isolated_spikes(level_series)
    s = pd.to_numeric(s, errors='coerce').where(lambda x: x > 0)
    return np.log(s.where(s.diff() != 0).dropna()).diff().reindex(s.index)


def load_panel() -> pd.DataFrame:
    """Daily index levels for all markets in the panel file."""
    return (pd.read_csv(config.PANEL_CSV, parse_dates=['date'], index_col='date')
              .sort_index())


def load_nepse() -> tuple[pd.Series, pd.Series | None]:
    """NEPSE closing level and turnover from the ShareSansar workbook."""
    nep = pd.read_excel(config.NEPSE_XLSX)
    nep = nep.rename(columns={c: c.strip() for c in nep.columns})
    nep = nep.set_index(pd.to_datetime(nep['Date'])).sort_index()
    close = pd.to_numeric(nep['Close'], errors='coerce')
    turnover = pd.to_numeric(nep['Turnover'], errors='coerce') if 'Turnover' in nep else None
    return close, turnover


def load_all() -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Panel, NEPSE close and NEPSE turnover in one call."""
    panel = load_panel()
    close, turnover = load_nepse()
    return panel, close, turnover


def coverage_table(panel: pd.DataFrame, nepse_close: pd.Series) -> pd.DataFrame:
    """Per-market coverage screen: last genuine observation and test-window coverage.

    Coverage is measured on genuine (pre-forward-fill) observations, which is
    what exposes series that quietly stop updating.
    """
    nep_gen = genuine_returns(nepse_close).dropna()
    test_days = nep_gen.index[(nep_gen.index >= config.TEST_START) &
                              (nep_gen.index <= config.SAMPLE_END)]
    rows = []
    for c in panel.columns:
        gen = genuine_level(panel[c])
        last = gen.last_valid_index()
        rows.append(dict(market=c,
                         last_obs=(last.date() if last is not None else None),
                         test_coverage=round(float(gen.reindex(test_days).notna().mean()), 3),
                         retained=('yes' if c in config.SELECTED else 'no')))
    return pd.DataFrame(rows).sort_values(['retained', 'market'], ascending=[False, True])


def lead_lag_correlations(panel: pd.DataFrame, lags=range(-5, 6)) -> pd.DataFrame:
    """Correlation of each foreign market's daily log return with NEPSE's, by lag.

    A positive lag means the foreign market leads NEPSE by that many rows.
    Levels are deliberately not used here: correlating non-stationary levels
    would be spurious.
    """
    markets = config.SELECTED + [config.NEPSE_PANEL_COL]
    returns = pd.DataFrame({c: np.log(remove_isolated_spikes(panel[c]).where(lambda x: x > 0))
                            .diff() * 100 for c in markets})
    returns = returns.sort_index().loc[config.TRAIN_START:config.SAMPLE_END]

    out = pd.DataFrame(index=config.SELECTED, columns=list(lags), dtype=float)
    nepse = returns[config.NEPSE_PANEL_COL]
    for c in config.SELECTED:
        for lag in lags:
            paired = pd.concat([returns[c].rename('foreign'),
                                nepse.shift(-lag).rename('nepse')], axis=1).dropna()
            if len(paired) >= 30:
                out.loc[c, lag] = paired['foreign'].corr(paired['nepse'])
    return out


def lead_lag_ranking(lag_corr: pd.DataFrame, lead_lags=(1, 2, 3, 4, 5)) -> pd.Series:
    """Sum of |correlation| over positive (leading) lags, ranked high to low."""
    return (lag_corr[list(lead_lags)].abs().sum(axis=1)
            .sort_values(ascending=False).rename('cumulative_abs_corr'))
