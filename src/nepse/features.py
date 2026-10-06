"""Feature engineering and assembly of the six information sets.

All predictors are lagged by at least one day, so a row dated *t* only ever
contains information available at the close of *t-1*.

Information sets
----------------
``NEPSE-only``      NEPSE's own return / momentum / volatility / turnover history
``Foreign-only``    raw lagged features of the 27 foreign markets
``Combined``        NEPSE + foreign + regional aggregates + calendar dummies
``GlobalPCA-only``  PCA factors of the foreign block
``NEPSE+PCA``       NEPSE history + PCA factors
``NEPSE+PLS``       NEPSE history + PLS (supervised) factors

``config.FIX_ROLLING`` controls how rolling windows are computed:

* ``True``  - windows run over each series' own trading days (dropna first),
  so a 5-day momentum really covers five trading days.
* ``False`` - windows run over the merged calendar, so gaps from other
  markets' holidays enter the window. This is the older behaviour and is kept
  only so the earlier results can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config
from .data import genuine_returns, load_all
from .models import median_imputer


@dataclass
class Dataset:
    """Everything the experiments need, built once."""
    info_sets: dict[str, pd.DataFrame]
    y: pd.Series
    train_mask: pd.Series
    panel: pd.DataFrame
    nepse_close: pd.Series
    turnover: pd.Series | None

    @property
    def y_train(self) -> pd.Series:
        return self.y[self.train_mask.values]

    @property
    def y_test(self) -> pd.Series:
        return self.y[~self.train_mask.values]


# --------------------------------------------------------------------------
# rolling / shifting helpers
# --------------------------------------------------------------------------
def _roll(series, window, how, fix_rolling):
    src = series.dropna() if fix_rolling else series
    r = src.rolling(window)
    v = r.sum() if how == 'sum' else r.std()
    return v.reindex(series.index)


def _shift(series, k, fix_rolling):
    src = series.dropna() if fix_rolling else series
    return src.shift(k).reindex(series.index)


# --------------------------------------------------------------------------
# feature blocks
# --------------------------------------------------------------------------
def nepse_features(nepse_ret, nepse_close, turnover=None, fix_rolling=True):
    f = pd.DataFrame(index=nepse_ret.index)
    for lag in (1, 2, 3, 5, 10):
        f[f'nep_ret_l{lag}'] = _shift(nepse_ret, lag, fix_rolling)
    for w in (5, 10, 20, 60):
        f[f'nep_mom_{w}'] = _shift(_roll(nepse_ret, w, 'sum', fix_rolling), 1, fix_rolling)
        f[f'nep_vol_{w}'] = _shift(_roll(nepse_ret, w, 'std', fix_rolling), 1, fix_rolling)

    close = nepse_close.reindex(nepse_ret.index)
    log_close = np.log(close.dropna() if fix_rolling else close)
    for w in (20, 60):
        f[f'nep_madist_{w}'] = ((log_close - log_close.rolling(w).mean())
                                .shift(1).reindex(nepse_ret.index))
    if turnover is not None:
        t = turnover.reindex(nepse_ret.index)
        f['nep_turn_l1'] = _shift(t, 1, fix_rolling)
        f['nep_turn_m5'] = _shift(_roll(t, 5, 'sum', fix_rolling) / 5, 1, fix_rolling)
    return f


def foreign_features(foreign_ret, base_lag=1, fix_rolling=True):
    cols = {}
    for c in config.SELECTED:
        r = foreign_ret[c]
        for extra in (0, 1, 2):
            cols[f'{c}_ret_l{base_lag + extra}'] = _shift(r, base_lag + extra, fix_rolling)
        cols[f'{c}_mom5'] = _shift(_roll(r, 5, 'sum', fix_rolling), base_lag, fix_rolling)
        cols[f'{c}_vol5'] = _shift(_roll(r, 5, 'std', fix_rolling), base_lag, fix_rolling)
    return pd.DataFrame(cols, index=foreign_ret.index)


def regional_features(foreign_ret, base_lag=1, fix_rolling=True):
    cols = {}
    for region, members in config.REGION.items():
        present = [c for c in members if c in foreign_ret.columns]
        agg = foreign_ret[present].mean(axis=1)
        cols[f'{region}_ret_l{base_lag}'] = _shift(agg, base_lag, fix_rolling)
        cols[f'{region}_mom5'] = _shift(_roll(agg, 5, 'sum', fix_rolling), base_lag, fix_rolling)
    return pd.DataFrame(cols, index=foreign_ret.index)


def calendar_features(idx):
    return pd.DataFrame({'dow': idx.dayofweek, 'month': idx.month}, index=idx)


def fit_factor_sets(x_foreign, y, train_mask, n_pca=10, n_pls=10):
    """PCA and PLS factors of the foreign block, fitted on training rows only."""
    x_train = x_foreign[train_mask]
    imp = median_imputer().fit(x_train)
    scaler = StandardScaler().fit(imp.transform(x_train))
    x_z = scaler.transform(imp.transform(x_foreign))

    pca = PCA(n_components=n_pca, random_state=config.RANDOM_STATE).fit(x_z[train_mask.values])
    pca_f = pd.DataFrame(pca.transform(x_z), index=x_foreign.index,
                         columns=[f'PC{i + 1}' for i in range(n_pca)])

    pls = PLSRegression(n_components=n_pls).fit(x_z[train_mask.values], y[train_mask].values)
    pls_f = pd.DataFrame(pls.transform(x_z), index=x_foreign.index,
                         columns=[f'PLS{i + 1}' for i in range(n_pls)])
    return pca_f, pls_f


def pca_explained_variance(ds: "Dataset", n_pca: int = 10):
    """Share of standardized foreign-feature variance captured by the PCA factors.

    Refits the same training-period imputer, scaler and PCA used in
    :func:`fit_factor_sets` and returns ``(first_component, all_components)``.
    """
    x_foreign, train_mask = ds.info_sets['Foreign-only'], ds.train_mask
    x_train = x_foreign[train_mask]
    imp = median_imputer().fit(x_train)
    scaler = StandardScaler().fit(imp.transform(x_train))
    x_z = scaler.transform(imp.transform(x_train))
    ratio = PCA(n_components=n_pca, random_state=config.RANDOM_STATE).fit(x_z).explained_variance_ratio_
    return float(ratio[0]), float(ratio.sum())


# --------------------------------------------------------------------------
# dataset assembly
# --------------------------------------------------------------------------
def build_dataset(fix_rolling: bool | None = None) -> Dataset:
    """Load the raw files and return the six aligned information sets."""
    fix_rolling = config.FIX_ROLLING if fix_rolling is None else fix_rolling

    panel, nepse_close, turnover = load_all()
    foreign_ret = pd.DataFrame({c: genuine_returns(panel[c]) for c in config.SELECTED})
    nepse_ret = genuine_returns(nepse_close)

    common = foreign_ret.index.union(nepse_ret.index)
    common = common[(common >= config.TRAIN_START) & (common <= config.SAMPLE_END)]
    foreign_ret = foreign_ret.reindex(common)
    nepse_ret = nepse_ret.reindex(common)
    nepse_close = nepse_close.reindex(common)
    turnover = turnover.reindex(common) if turnover is not None else None

    nep = nepse_features(nepse_ret, nepse_close, turnover, fix_rolling)
    forn = foreign_features(foreign_ret, fix_rolling=fix_rolling)
    reg = regional_features(foreign_ret, fix_rolling=fix_rolling)
    cal = calendar_features(common)

    y = nepse_ret.copy()
    rows = y.notna()
    for block in (nep, forn, reg, cal):
        rows &= block.notna().any(axis=1)
    y = y[rows]
    nep, forn, reg, cal = [b.loc[y.index] for b in (nep, forn, reg, cal)]

    train_mask = pd.Series(y.index < config.TEST_START, index=y.index)
    pca_f, pls_f = fit_factor_sets(forn, y, train_mask)
    pca_f, pls_f = pca_f.loc[y.index], pls_f.loc[y.index]

    info_sets = {
        'NEPSE-only':     nep,
        'NEPSE+PCA':      pd.concat([nep, pca_f], axis=1),
        'NEPSE+PLS':      pd.concat([nep, pls_f], axis=1),
        'GlobalPCA-only': pca_f,
        'Foreign-only':   forn,
        'Combined':       pd.concat([nep, forn, reg, cal], axis=1),
    }
    return Dataset(info_sets=info_sets, y=y, train_mask=train_mask, panel=panel,
                   nepse_close=nepse_close, turnover=turnover)
