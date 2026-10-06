"""Figures. Each figure is defined once here and drawn once by a notebook.

Every function saves to ``results/figures/`` as PNG and PDF and returns the
matplotlib figure. File names follow the figure numbers in the paper
(Figure 2, the methodology diagram, is not generated from data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config
from .data import genuine_level

MODEL_NAMES = {'OLS': 'OLS', 'Ridge': 'Ridge', 'RF': 'Random Forest', 'ET': 'Extra Trees',
               'GBR': 'Gradient Boosting', 'XGB': 'XGBoost', 'LSTM': 'LSTM', 'GRU': 'GRU'}


def apply_style() -> None:
    """Publication defaults; call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 11,
        "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "legend.frameon": False, "figure.autolayout": False,
    })


def save_figure(fig, name: str) -> None:
    config.ensure_dirs()
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURE_DIR / f"{name}.{ext}", bbox_inches="tight")
    print(f"[saved] figures/{name}.png / .pdf")


# --------------------------------------------------------------------------
# Figure 1 - market overview (notebook 01)
# --------------------------------------------------------------------------
def market_overview(panel: pd.DataFrame, nepse_close: pd.Series,
                    name: str = "fig1_market_overview"):
    """Panel A: rebased index levels with the test window shaded.
    Panel B: daily log returns. NEPSE against the 27 retained peers.
    """
    fig, (ax_level, ax_ret) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    peer_labelled = False

    for market in config.SELECTED:
        s = genuine_level(panel[market]).dropna().loc[config.TRAIN_START:config.SAMPLE_END]
        if len(s) < 100:
            continue
        ax_level.plot(s.index, s / s.iloc[0], color=config.PEER_COLOR, lw=0.65, alpha=0.22,
                      zorder=2, label=None if peer_labelled else f'Peers (N = {len(config.SELECTED)})')
        ax_ret.plot(s.index, np.log(s).diff() * 100, color=config.PEER_COLOR, lw=0.45,
                    alpha=0.18, zorder=2)
        peer_labelled = True

    nepse = genuine_level(nepse_close).dropna().loc[config.TRAIN_START:config.SAMPLE_END]
    ax_level.plot(nepse.index, nepse / nepse.iloc[0], color=config.NEPSE_COLOR, lw=2.0,
                  zorder=5, label='NEPSE')
    ax_ret.plot(nepse.index, np.log(nepse).diff() * 100, color=config.NEPSE_COLOR, lw=1.1,
                zorder=5, label='NEPSE')

    ax_level.axvspan(config.TEST_START, config.SAMPLE_END, color=config.ACCENT_COLOR, alpha=0.08)
    ax_level.text(config.TEST_START, ax_level.get_ylim()[1] * 0.96, "  test period",
                  color=config.ACCENT_COLOR, fontsize=9, va='top')
    ax_level.set_yscale('log')
    ax_level.set_ylabel('Index, rebased to 1.0 at start (log scale)')
    ax_level.set_title('A. Index levels')
    ax_level.legend(loc='upper left')

    ax_ret.axhline(0, color='black', lw=0.7, alpha=0.7)
    ax_ret.set_ylabel('Daily log return (%)')
    ax_ret.set_title('B. Daily log returns')
    ax_ret.set_xlabel('Year')
    ax_level.set_xlabel('Year')

    fig.tight_layout()
    save_figure(fig, name)
    return fig


# --------------------------------------------------------------------------
# Figure 3 - lead-lag structure (notebook 01)
# --------------------------------------------------------------------------
def leadlag_heatmap(lag_corr: pd.DataFrame, name: str = "fig3_leadlag_heatmap"):
    """Correlation of each foreign market's return with NEPSE's, by lag.

    Positive lags: the foreign market leads NEPSE.
    """
    limit = max(0.10, float(np.nanmax(np.abs(lag_corr.to_numpy()))))
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(lag_corr.values.astype(float), cmap='RdBu_r', vmin=-limit, vmax=limit,
                   aspect='auto')
    ax.set_xticks(range(lag_corr.shape[1]))
    ax.set_xticklabels(lag_corr.columns)
    ax.set_yticks(range(lag_corr.shape[0]))
    ax.set_yticklabels(lag_corr.index)
    for i in range(lag_corr.shape[0]):
        for j in range(lag_corr.shape[1]):
            v = lag_corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
    zero = list(lag_corr.columns).index(0)
    ax.axvline(zero - 0.5, color='black', lw=1.0)
    ax.axvline(zero + 0.5, color='black', lw=1.0)
    ax.set_xlabel('Lag (days)')
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label='Correlation')
    fig.tight_layout()
    save_figure(fig, name)
    return fig


# --------------------------------------------------------------------------
# Figures 4 & 5 - metric heatmaps (notebook 04)
# --------------------------------------------------------------------------
def metric_heatmap(matrix: pd.DataFrame, title: str, name: str, fmt: str = "{:.1f}",
                   lower_better: bool = True):
    """Models x information sets heatmap (RMSE or directional accuracy)."""
    disp = matrix.rename(columns=config.INFO_LABELS)
    fig, ax = plt.subplots(figsize=(1.8 + 1.35 * disp.shape[1], 1.4 + 0.55 * disp.shape[0]))
    cmap = LinearSegmentedColormap.from_list(
        "bg", ["#2c7fb8", "#ffffbf", "#d7191c"] if lower_better
        else ["#d7191c", "#ffffbf", "#2c7fb8"])
    im = ax.imshow(disp.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(disp.shape[1]))
    ax.set_xticklabels(disp.columns, rotation=30, ha='right')
    ax.set_yticks(range(disp.shape[0]))
    ax.set_yticklabels(disp.index)
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            v = disp.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=9)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.82)
    save_figure(fig, name)
    return fig


# --------------------------------------------------------------------------
# Figure 6 - predicted vs observed (notebook 04)
# --------------------------------------------------------------------------
def pred_vs_obs(dates, actual, pred, model_name: str, info_label: str, model_metrics: dict,
                name: str | None = None):
    """Two panels: the test-period series, and predicted against observed."""
    a = np.asarray(actual).ravel() * config.RET_SCALE
    p = np.asarray(pred).ravel() * config.RET_SCALE

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={'width_ratios': [1.9, 1]})
    ax[0].plot(dates, a, color=config.OBS_COLOR, lw=0.9, label='Observed')
    ax[0].plot(dates, p, color=config.PRED_COLOR, lw=0.9, alpha=0.85, label='Predicted')
    ax[0].axhline(0, color='grey', lw=0.6)
    ax[0].set_ylabel('Daily return (bps)')
    ax[0].set_title(f'{MODEL_NAMES.get(model_name, model_name)} on {info_label}: '
                    'observed and predicted (test period)')
    ax[0].legend(loc='upper right')
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    lim = float(np.nanpercentile(np.abs(np.concatenate([a, p])), 99))
    ax[1].scatter(a, p, s=8, alpha=0.35, color=config.OBS_COLOR, edgecolors='none')
    ax[1].plot([-lim, lim], [-lim, lim], color=config.PRED_COLOR, ls='--', lw=1.2)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel('Observed (bps)')
    ax[1].set_ylabel('Predicted (bps)')
    ax[1].set_title('Predicted vs observed')
    ax[1].text(0.04, 0.96,
               f"R\u00b2 = {model_metrics['R2_vs_mean']:.3f}\n"
               f"Corr = {model_metrics['Corr']:.3f}\n"
               f"DirAcc = {model_metrics['DirAcc']:.3f}",
               transform=ax[1].transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    fig.tight_layout()
    save_figure(fig, name or f"fig6_pred_vs_obs_{model_name}")
    return fig
