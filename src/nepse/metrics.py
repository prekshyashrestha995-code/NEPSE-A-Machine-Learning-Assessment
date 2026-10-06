"""Forecast evaluation metrics and tests.

Returns are handled in log units internally and reported in basis points
(``config.RET_SCALE``).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config


def rmse(actual, pred) -> float:
    a, p = np.asarray(actual).ravel(), np.asarray(pred).ravel()
    return float(np.sqrt(np.mean((a - p) ** 2)))


def dir_accuracy(actual, pred) -> float:
    a, p = np.asarray(actual).ravel(), np.asarray(pred).ravel()
    return float(np.mean(np.sign(a) == np.sign(p)))


def pesaran_timmermann(actual, pred) -> float:
    """p-value of the Pesaran-Timmermann test of directional predictability."""
    a, p = np.asarray(actual).ravel(), np.asarray(pred).ravel()
    x, yhat = (a > 0).astype(float), (p > 0).astype(float)
    n = len(a)
    py, px = x.mean(), yhat.mean()
    phat = np.mean(x == yhat)
    pstar = py * px + (1 - py) * (1 - px)
    var = (pstar * (1 - pstar) / n
           - (((2 * py - 1) ** 2) * px * (1 - px) / n
              + ((2 * px - 1) ** 2) * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2))
    if var <= 0:
        return float('nan')
    return float(2 * (1 - stats.norm.cdf(abs((phat - pstar) / np.sqrt(var)))))


def evaluate(actual, pred, train_mean) -> dict:
    """RMSE, MAE, out-of-sample R2, directional accuracy, PT p-value, correlation."""
    a, p = np.asarray(actual).ravel(), np.asarray(pred).ravel()
    bench = np.full(len(a), train_mean)
    sse, ssb = np.sum((a - p) ** 2), np.sum((a - bench) ** 2)
    return {
        'RMSE_bps':   rmse(a, p) * config.RET_SCALE,
        'MAE_bps':    float(np.mean(np.abs(a - p))) * config.RET_SCALE,
        'R2_vs_mean': float(1 - sse / ssb) if ssb > 0 else float('nan'),
        'DirAcc':     dir_accuracy(a, p),
        'PT_p':       pesaran_timmermann(a, p),
        'Corr':       float(np.corrcoef(a, p)[0, 1]),
    }


def diebold_mariano(errors_1, errors_2, h=1, loss='squared'):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    Returns ``(statistic, p_value, favoured)`` where ``favoured`` is ``'model1'``
    when the first error series has the lower average loss.
    """
    e1, e2 = np.asarray(errors_1).ravel(), np.asarray(errors_2).ravel()
    d = (e1 ** 2 - e2 ** 2) if loss == 'squared' else (np.abs(e1) - np.abs(e2))
    n = len(d)
    dbar = d.mean()
    lrv = np.mean((d - dbar) ** 2) + 2 * sum(
        np.mean((d[k:] - dbar) * (d[:-k] - dbar)) for k in range(1, h))
    if lrv <= 0:
        return float('nan'), float('nan'), 'NA'
    stat = dbar / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    return float(stat), p, ('model1' if dbar < 0 else 'model2')


def zero_return_rmse_bps(actual) -> float:
    """RMSE of the null forecast (always predict a zero return), in bps."""
    a = np.asarray(actual).ravel()
    return rmse(a, np.zeros(len(a))) * config.RET_SCALE
