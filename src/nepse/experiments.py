"""Experiment drivers and result storage.

Every model is fitted exactly once, in :func:`run_classical_grid` (or
:func:`run_deep_learning`). The results are written to ``results/tables/`` as
long-form metrics plus a matrix of test-set predictions, and every figure and
summary table is built from those files. Nothing downstream refits a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, metrics as M
from .features import Dataset
from .models import model_registry, tune_fit

METRIC_COLS = ['RMSE_bps', 'MAE_bps', 'R2_vs_mean', 'DirAcc', 'PT_p', 'Corr']

_METRICS_CLASSICAL = "metrics_classical.csv"
_PREDS_CLASSICAL = "predictions_classical.csv"
_METRICS_DEEP = "metrics_deep.csv"
_PREDS_DEEP = "predictions_deep.csv"
_WALKFORWARD = "walkforward.csv"


# --------------------------------------------------------------------------
# classical grid
# --------------------------------------------------------------------------
def run_classical_grid(ds: Dataset, info_order=None, extended: bool | None = None,
                       verbose: bool = True):
    """Fit every classical model on every information set once.

    Returns ``(metrics_long, predictions)`` where ``predictions`` is indexed by
    test date with one column per ``"<info>|<model>"`` plus ``observed``.
    """
    extended = config.EXTENDED_SEARCH if extended is None else extended
    info_order = info_order or config.INFO_ORDER
    registry = model_registry(extended)

    y_train, y_test = ds.y_train, ds.y_test
    actual = y_test.values
    records = {}
    predictions = {'observed': pd.Series(actual, index=y_test.index)}

    for info in info_order:
        x = ds.info_sets[info]
        x_train, x_test = x[ds.train_mask.values], x[~ds.train_mask.values]
        for name, spec in registry.items():
            try:
                fitted, params = tune_fit(spec['pipe'], spec['grid'], x_train, y_train,
                                          n_splits=3 if extended else 2)
                pred = fitted.predict(x_test)
            except Exception as exc:                       # keep the grid going
                if verbose:
                    print(f"  [skip] {info} | {name}: {exc}")
                continue
            records[(info, name)] = dict(
                info=info, model=name, family='classical',
                **M.evaluate(actual, pred, y_train.mean()),
                params=str(params) if params else 'defaults')
            predictions[f'{info}|{name}'] = pd.Series(pred, index=y_test.index)
            if verbose:
                print(f"  {info:<15} {name:<6} RMSE {records[(info, name)]['RMSE_bps']:8.2f} bps")

    metrics_long = pd.DataFrame(list(records.values()))
    return metrics_long, pd.DataFrame(predictions)


# --------------------------------------------------------------------------
# walk-forward stability
# --------------------------------------------------------------------------
def run_walk_forward(ds: Dataset, model: str | None = None, n_origins: int | None = None,
                     info_order=None, extended: bool | None = None) -> pd.DataFrame:
    """Re-estimate one model at several expanding origins; report RMSE mean and s.d.

    The test period is cut into ``n_origins + 1`` equal blocks. At origin k the
    model is refitted on every day before block k and scored on block k, so the
    six origins of the paper cover the first six of seven blocks.
    """
    extended = config.EXTENDED_SEARCH if extended is None else extended
    info_order = info_order or config.INFO_ORDER
    model = model or config.WALKFORWARD_MODEL
    n_origins = n_origins if n_origins is not None else config.WALKFORWARD_ORIGINS

    spec = model_registry(extended)[model]
    idx = ds.y.index
    start = np.where(~ds.train_mask.values)[0][0]
    block = max(20, (len(idx) - start) // (n_origins + 1))

    rows = []
    for origin in range(n_origins):
        cut = start + origin * block
        if cut + block > len(idx):
            break
        train_idx, test_idx = idx[:cut], idx[cut:cut + block]
        for info in info_order:
            x = ds.info_sets[info]
            try:
                fitted, _ = tune_fit(spec['pipe'], spec['grid'], x.loc[train_idx],
                                     ds.y.loc[train_idx], n_splits=2)
                pred = fitted.predict(x.loc[test_idx])
            except Exception:
                continue
            rows.append(dict(origin=origin, info=info,
                             RMSE_bps=M.rmse(ds.y.loc[test_idx].values, pred) * config.RET_SCALE))

    wf = pd.DataFrame(rows)
    if wf.empty:
        return pd.DataFrame(columns=['mean', 'std'])
    return wf.groupby('info')['RMSE_bps'].agg(['mean', 'std']).reindex(info_order)


# --------------------------------------------------------------------------
# deep learning
# --------------------------------------------------------------------------
def run_deep_learning(ds: Dataset, cells=('LSTM', 'GRU'), info_order=None,
                      verbose: bool = True):
    """Train one sequence model per cell type and information set."""
    from . import deep

    info_order = info_order or config.INFO_ORDER
    cfg = deep.dl_config()
    y_train = ds.y_train
    records, predictions = [], {}

    for info in info_order:
        x = ds.info_sets[info]
        seq_x, seq_y, dates, is_train = deep.prepare_sequences(
            x, ds.y, ds.train_mask, config.LOOKBACK)
        x_tr, y_tr = seq_x[is_train], seq_y[is_train]
        x_te, y_te = seq_x[~is_train], seq_y[~is_train]
        test_dates = dates[~is_train]
        n_features = seq_x.shape[2]

        for cell in cells:
            model = deep.build_seq_model(n_features, config.LOOKBACK, cell, cfg)
            mu, sd = float(y_tr.mean()), float(y_tr.std() + 1e-12)
            model, epochs = deep.fit_seq_model(model, x_tr, (y_tr - mu) / sd, cfg)
            pred = model.predict(x_te, verbose=0).ravel() * sd + mu

            records.append(dict(info=info, model=cell, family='deep',
                                **M.evaluate(y_te, pred, y_train.mean()),
                                params=str(dict(lookback=config.LOOKBACK, epochs=epochs, **cfg))))
            predictions[f'{info}|{cell}'] = pd.Series(pred, index=test_dates)
            if verbose:
                print(f"  {info:<15} {cell:<6} RMSE {records[-1]['RMSE_bps']:8.2f} bps "
                      f"(stopped at epoch {epochs})")

    preds = pd.DataFrame(predictions)
    preds.insert(0, 'observed', ds.y.reindex(preds.index))
    return pd.DataFrame(records), preds


# --------------------------------------------------------------------------
# summaries built from stored results
# --------------------------------------------------------------------------
def metric_matrix(metrics_long: pd.DataFrame, metric: str = 'RMSE_bps',
                  info_order=None, model_order=None) -> pd.DataFrame:
    """Long-form metrics -> models x information sets matrix."""
    info_order = [i for i in (info_order or config.INFO_ORDER)
                  if i in set(metrics_long['info'])]
    mat = metrics_long.pivot(index='model', columns='info', values=metric)[info_order]
    model_order = model_order or config.MODEL_ORDER
    ordered = [m for m in model_order if m in mat.index]
    ordered += [m for m in mat.index if m not in ordered]     # anything unexpected, last
    return mat.reindex(ordered).astype(float)


def dm_table(predictions: pd.DataFrame, model: str = 'ET', base_info: str = 'NEPSE-only',
             info_order=None) -> pd.DataFrame:
    """Diebold-Mariano tests of each information set against ``base_info``, same model."""
    info_order = info_order or config.INFO_ORDER
    base_key = f'{base_info}|{model}'
    if base_key not in predictions:
        return pd.DataFrame()

    observed = predictions['observed'].values
    base_err = observed - predictions[base_key].values
    rows = []
    for info in info_order:
        key = f'{info}|{model}'
        if info == base_info or key not in predictions:
            continue
        err = observed - predictions[key].values
        for loss in ('squared', 'absolute'):
            stat, p, favoured = M.diebold_mariano(base_err, err, loss=loss)
            rows.append(dict(information_set=config.INFO_LABELS[info], model=model, loss=loss,
                             DM=stat, p=p,
                             favours=(config.INFO_LABELS[base_info] if favoured == 'model1'
                                      else config.INFO_LABELS[info])))
    return pd.DataFrame(rows)


def best_per_set(metrics_long: pd.DataFrame, predictions: pd.DataFrame,
                 dm_model: str = 'ET', info_order=None) -> pd.DataFrame:
    """Best model per information set, with lift over NEPSE-only and DM p-values."""
    info_order = [i for i in (info_order or config.INFO_ORDER)
                  if i in set(metrics_long['info'])]
    rmse_mat = metric_matrix(metrics_long, 'RMSE_bps', info_order)
    dir_mat = metric_matrix(metrics_long, 'DirAcc', info_order)

    best = pd.DataFrame({'best_model': rmse_mat.idxmin(),
                         'RMSE_bps': rmse_mat.min(),
                         'best_DirAcc_any_model': dir_mat.max()}).reindex(info_order)
    nepse_rmse = best.loc['NEPSE-only', 'RMSE_bps']
    base = M.zero_return_rmse_bps(predictions['observed'].values)
    best['lift_vs_NEPSE_%'] = (nepse_rmse - best['RMSE_bps']) / nepse_rmse * 100
    best['vs_zero_%'] = (base - best['RMSE_bps']) / base * 100

    dm = dm_table(predictions, model=dm_model, info_order=info_order)
    p_by_label = (dm[dm['loss'] == 'squared'].set_index('information_set')['p'].to_dict()
                  if len(dm) else {})
    best[f'DM_p_vs_NEPSE_{dm_model}'] = [p_by_label.get(config.INFO_LABELS[i], np.nan)
                                         for i in best.index]
    best.index = [config.INFO_LABELS[i] for i in best.index]
    return best.round(3)


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def save_classical(metrics_long, predictions, walkforward=None) -> None:
    config.ensure_dirs()
    metrics_long.to_csv(config.TABLE_DIR / _METRICS_CLASSICAL, index=False)
    predictions.to_csv(config.TABLE_DIR / _PREDS_CLASSICAL)
    if walkforward is not None:
        walkforward.to_csv(config.TABLE_DIR / _WALKFORWARD)
    print(f"[saved] classical results -> {config.TABLE_DIR}")


def save_deep(metrics_long, predictions) -> None:
    config.ensure_dirs()
    metrics_long.to_csv(config.TABLE_DIR / _METRICS_DEEP, index=False)
    predictions.drop(columns=['observed']).to_csv(config.TABLE_DIR / _PREDS_DEEP)
    print(f"[saved] deep-learning results -> {config.TABLE_DIR}")


def load_results(include_deep: bool = True):
    """Load stored results. Returns ``(metrics_long, predictions, walkforward)``."""
    path = config.TABLE_DIR / _METRICS_CLASSICAL
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run notebook 02 (model comparison) first.")

    metrics_long = pd.read_csv(path)
    predictions = pd.read_csv(config.TABLE_DIR / _PREDS_CLASSICAL,
                              index_col=0, parse_dates=True)

    deep_metrics = config.TABLE_DIR / _METRICS_DEEP
    if include_deep and deep_metrics.exists():
        metrics_long = pd.concat([metrics_long, pd.read_csv(deep_metrics)], ignore_index=True)
        deep_preds = pd.read_csv(config.TABLE_DIR / _PREDS_DEEP, index_col=0, parse_dates=True)
        predictions = predictions.join(deep_preds, how='left')

    wf_path = config.TABLE_DIR / _WALKFORWARD
    walkforward = (pd.read_csv(wf_path, index_col=0) if wf_path.exists() else None)
    return metrics_long, predictions, walkforward
