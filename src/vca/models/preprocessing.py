"""Baseline-covariate preprocessing into a numeric design matrix.

Wraps a scikit-learn :class:`~sklearn.compose.ColumnTransformer` so that models
receive a clean, standardised design matrix regardless of which optional
covariates a given dataset happens to carry. Missingness is handled explicitly
and, for covariates whose missingness is plausibly informative (e.g. PD-L1 not
tested in older comparator arms), a missing-indicator column is added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Candidate covariate roles for NSCLC. Only those actually present in the input
# are used, so datasets missing e.g. biomarker columns still work.
DEFAULT_NUMERIC = [
    "age",
    "ecog_ps",           # treated as an ordinal numeric score
    "prior_lines",
    "baseline_sld_mm",
    "n_target_lesions",
    "pdl1_tps",
]
DEFAULT_CATEGORICAL = [
    "sex",
    "stage",
    "histology",
    "smoking",
    "egfr_status",
    "alk_status",
]
# Numeric columns whose missingness is itself informative -> add indicator.
INFORMATIVE_MISSING = ["pdl1_tps"]


class CovariatePreprocessor:
    """Fit/transform baseline covariates into a model-ready design matrix.

    Examples
    --------
    >>> pre = CovariatePreprocessor()
    >>> X = pre.fit_transform(trial.baseline)      # doctest: +SKIP
    >>> pre.feature_names_[:3]                      # doctest: +SKIP
    ['num__age', 'num__ecog_ps', 'num__prior_lines']
    """

    def __init__(
        self,
        numeric: list[str] | None = None,
        categorical: list[str] | None = None,
    ) -> None:
        self._numeric_req = list(numeric) if numeric is not None else DEFAULT_NUMERIC
        self._categorical_req = (
            list(categorical) if categorical is not None else DEFAULT_CATEGORICAL
        )
        self.column_transformer_: ColumnTransformer | None = None
        self.feature_names_: list[str] = []
        self.numeric_: list[str] = []
        self.categorical_: list[str] = []

    def _build(self, df: pd.DataFrame) -> ColumnTransformer:
        self.numeric_ = [c for c in self._numeric_req if c in df.columns]
        self.categorical_ = [c for c in self._categorical_req if c in df.columns]
        if not self.numeric_ and not self.categorical_:
            raise ValueError("No usable covariate columns found in the baseline frame.")

        add_indicator = any(c in self.numeric_ for c in INFORMATIVE_MISSING)
        numeric_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
                ("scale", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers = []
        if self.numeric_:
            transformers.append(("num", numeric_pipe, self.numeric_))
        if self.categorical_:
            transformers.append(("cat", categorical_pipe, self.categorical_))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, df: pd.DataFrame) -> "CovariatePreprocessor":
        self.column_transformer_ = self._build(df)
        self.column_transformer_.fit(df)
        self.feature_names_ = list(self.column_transformer_.get_feature_names_out())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.column_transformer_ is None:
            raise RuntimeError("CovariatePreprocessor must be .fit() before .transform()")
        X = self.column_transformer_.transform(df)
        return np.asarray(X, dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
