"""Narrow, offline scikit-learn compatibility for the unmodified course.

Legacy Boston data is shipped from scikit-learn 1.1.3, not fetched at runtime.
The original dataset was deprecated for well-documented ethical concerns;
this adapter exists only so the historical course notebook remains executable.
"""
from __future__ import annotations

from pathlib import Path


def install_boston():
    import numpy as np
    import sklearn.datasets as datasets
    from sklearn.utils import Bunch

    if "load_boston" in datasets.__dict__:
        return

    csv_file = Path(__file__).parent / "legacy_data" / "boston_house_prices.csv"
    features = np.array([
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ])
    def load_boston(*, return_X_y=False):
        values = np.loadtxt(csv_file, delimiter=",", skiprows=2, dtype=np.float64)
        if values.shape != (506, 14):
            raise RuntimeError(f"Invalid bundled Boston dataset: {values.shape}")
        data, target = values[:, :13].copy(), values[:, 13].copy()
        if return_X_y:
            return data, target
        return Bunch(
            data=data, target=target, feature_names=features.copy(),
            filename=str(csv_file),
            DESCR="Historical Boston house-prices dataset; kept solely for legacy coursework."
        )
    datasets.load_boston = load_boston


def install_linear_regression():
    import inspect
    import sklearn.linear_model as linear
    Original = linear.LinearRegression
    if getattr(Original, "_ml_legacy_normalize", False):
        return

    class LinearRegression(Original):
        """scikit-learn LinearRegression with deprecated normalize=False accepted."""

        _ml_legacy_normalize = True

        def __init__(self, *, fit_intercept=True, copy_X=True, tol=1e-6,
                     n_jobs=None, positive=False, normalize=False):
            if normalize not in (False, None):
                raise ValueError(
                    "normalize=True is not supported; use a preprocessing scaler"
                )
            self.normalize = normalize
            self.tol = tol
            kwargs = dict(fit_intercept=fit_intercept, copy_X=copy_X,
                          n_jobs=n_jobs, positive=positive)
            if "tol" in inspect.signature(Original.__init__).parameters:
                kwargs["tol"] = tol
            super().__init__(**kwargs)

    LinearRegression.__name__ = "LinearRegression"
    LinearRegression.__qualname__ = "LinearRegression"
    LinearRegression.__module__ = "sklearn.linear_model._base"
    import sklearn.linear_model._base as base
    base.LinearRegression = LinearRegression
    linear.LinearRegression = LinearRegression
