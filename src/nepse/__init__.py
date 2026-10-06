"""Shared code for the NEPSE global-markets forecasting study.

Every notebook imports from here, so the data cleaning, feature engineering,
model definitions and metrics exist in exactly one place.

Typical use inside a notebook:

    from nepse import config, data, features, experiments, plots
    ds = features.build_dataset()            # defaults reproduce the paper
"""

from . import config, data, features, metrics, models, experiments, plots  # noqa: F401

__all__ = ["config", "data", "features", "metrics", "models", "experiments", "plots"]
