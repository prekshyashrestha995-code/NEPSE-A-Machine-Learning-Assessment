"""LSTM / GRU sequence models.

Kept in a separate module so the rest of the package imports without
TensorFlow installed. Import this only when running notebook 03.

Leakage controls: imputation and scaling are fitted on training rows only,
the split is chronological, and early stopping validates on a chronological
tail of the training window.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .models import median_imputer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _tf():
    import tensorflow as tf
    return tf


def set_seeds(seed: int | None = None) -> None:
    seed = config.RANDOM_STATE if seed is None else seed
    np.random.seed(seed)
    _tf().random.set_seed(seed)


def dl_config() -> dict:
    """Deep-learning hyperparameters. The paper uses ``config.DL_CONFIG`` (8 epochs)."""
    cfg = dict(config.DL_CONFIG)
    if config.EXTENDED_SEARCH:
        cfg['max_epochs'] = config.DL_MAX_EPOCHS_EXTENDED
    return cfg


def prepare_sequences(x: pd.DataFrame, y: pd.Series, train_mask: pd.Series, lookback: int):
    """Impute+scale on training rows, then build ``[samples, lookback, features]``."""
    imp = median_imputer().fit(x[train_mask])
    x_imputed = imp.transform(x)
    scaler = StandardScaler().fit(x_imputed[train_mask.values])
    x_scaled = scaler.transform(x_imputed)

    idx = y.index
    seq_x, seq_y, dates = [], [], []
    for i in range(lookback - 1, len(idx)):
        seq_x.append(x_scaled[i - lookback + 1:i + 1])
        seq_y.append(y.iloc[i])
        dates.append(idx[i])

    seq_x = np.asarray(seq_x, dtype='float32')
    seq_y = np.asarray(seq_y, dtype='float32')
    dates = pd.DatetimeIndex(dates)
    return seq_x, seq_y, dates, dates < config.TEST_START


def build_seq_model(n_features: int, lookback: int, cell: str = 'LSTM', cfg: dict | None = None):
    tf = _tf()
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    cfg = cfg or dl_config()
    set_seeds()
    model = Sequential([Input((lookback, n_features)),
                        (LSTM if cell == 'LSTM' else GRU)(cfg['units']),
                        Dropout(cfg['dropout']),
                        Dense(1)])
    model.compile(optimizer=Adam(cfg['lr']), loss='mse')
    return model


def fit_seq_model(model, x_train, y_train, cfg: dict | None = None):
    from tensorflow.keras.callbacks import EarlyStopping

    cfg = cfg or dl_config()
    n_val = max(1, int(len(x_train) * cfg['val_frac']))          # chronological tail
    x_fit, y_fit = x_train[:-n_val], y_train[:-n_val]
    x_val, y_val = x_train[-n_val:], y_train[-n_val:]
    stopper = EarlyStopping(monitor='val_loss', patience=cfg['patience'],
                            restore_best_weights=True)
    history = model.fit(x_fit, y_fit, validation_data=(x_val, y_val),
                        epochs=cfg['max_epochs'], batch_size=cfg['batch'],
                        verbose=0, callbacks=[stopper], shuffle=True)
    return model, len(history.history['loss'])
