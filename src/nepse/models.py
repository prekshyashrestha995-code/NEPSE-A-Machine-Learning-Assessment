"""Classical model registry and time-series hyperparameter search.

Six models: OLS, Ridge, Random Forest, Extra Trees, Gradient Boosting and
XGBoost (skipped automatically if xgboost is not installed).

Imputation and scaling sit inside each pipeline, so they are refitted on the
training rows of every CV fold rather than on the whole sample.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:                                    # pragma: no cover
    HAS_XGB = False


def median_imputer() -> SimpleImputer:
    """Median imputer that keeps all-NaN columns (so feature counts stay stable)."""
    try:
        return SimpleImputer(strategy='median', keep_empty_features=True)
    except TypeError:                                # older scikit-learn
        return SimpleImputer(strategy='median')


def _scaled(estimator) -> Pipeline:
    return Pipeline([('imp', median_imputer()), ('sc', StandardScaler()), ('m', estimator)])


def _raw(estimator) -> Pipeline:
    return Pipeline([('imp', median_imputer()), ('m', estimator)])


def model_registry(extended: bool | None = None) -> dict:
    """``{name: {'pipe':..., 'grid':..., 'kind':...}}`` for every classical model.

    With ``extended=False`` (the default, and the paper's setting) each grid is
    a single configuration except Ridge's alpha in {1, 10}.
    """
    extended = config.EXTENDED_SEARCH if extended is None else extended
    s = not extended                                   # s = small (paper) grids
    rs = config.RANDOM_STATE

    registry = {
        'OLS': dict(pipe=_scaled(LinearRegression()), grid={}, kind='Linear (OLS)'),
        'Ridge': dict(pipe=_scaled(Ridge(random_state=rs)),
                      grid={'m__alpha': [1.0, 10.0] if s else [0.1, 1, 10, 100]},
                      kind='Linear (L2)'),
        'RF': dict(pipe=_raw(RandomForestRegressor(random_state=rs, n_jobs=-1)),
                   grid={'m__n_estimators': [200] if s else [300, 600],
                         'm__max_depth': [6] if s else [4, 8, None]},
                   kind='Random Forest'),
        'ET': dict(pipe=_raw(ExtraTreesRegressor(random_state=rs, n_jobs=-1)),
                   grid={'m__n_estimators': [200] if s else [300, 600],
                         'm__max_depth': [6] if s else [4, 8, None]},
                   kind='Extra Trees'),
        'GBR': dict(pipe=_raw(GradientBoostingRegressor(random_state=rs)),
                    grid={'m__n_estimators': [200] if s else [300, 600],
                          'm__learning_rate': [0.05] if s else [0.03, 0.05, 0.1],
                          'm__max_depth': [3] if s else [2, 3]},
                    kind='Gradient Boosting'),
    }
    if HAS_XGB:
        registry['XGB'] = dict(
            pipe=_raw(XGBRegressor(random_state=rs, n_jobs=-1, objective='reg:squarederror')),
            grid={'m__n_estimators': [300] if s else [400, 800],
                  'm__learning_rate': [0.05] if s else [0.03, 0.05, 0.1],
                  'm__max_depth': [3] if s else [3, 5],
                  'm__subsample': [0.9]},
            kind='XGBoost')
    return registry


def tune_fit(pipe, grid, x_train, y_train, n_splits=3):
    """Grid search scored by expanding-window CV RMSE; refit on all training rows.

    Returns ``(fitted_pipeline, selected_params)``.
    """
    if not grid:
        return clone(pipe).fit(x_train, y_train), {}

    keys = list(grid)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params, best_score = None, np.inf
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        errs = []
        for tr, va in tscv.split(x_train):
            model = clone(pipe).set_params(**params).fit(x_train.iloc[tr], y_train.iloc[tr])
            errs.append(np.sqrt(np.mean(
                (y_train.iloc[va].values - model.predict(x_train.iloc[va])) ** 2)))
        score = float(np.mean(errs))
        if score < best_score:
            best_score, best_params = score, params

    fitted = clone(pipe).set_params(**best_params).fit(x_train, y_train)
    return fitted, {k.replace('m__', ''): v for k, v in best_params.items()}
