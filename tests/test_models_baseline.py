import numpy as np
import pytest

from vca.models.base import SimulationResult
from vca.models.baseline import MarginalResamplingModel


def test_fit_returns_self(split):
    train, _ = split
    m = MarginalResamplingModel()
    assert m.fit(train) is m
    assert m.is_fitted


def test_simulate_requires_fit(split):
    _, test = split
    with pytest.raises(RuntimeError):
        MarginalResamplingModel().simulate(test.covariates())


def test_simulate_shapes_and_ranges(split):
    train, test = split
    m = MarginalResamplingModel().fit(train)
    times = np.arange(0.0, 361.0, 30.0)
    res = m.simulate(test.covariates(), n_draws=50, times=times, seed=1)
    assert isinstance(res, SimulationResult)
    assert res.n_draws == 50
    assert res.n_patients == test.n_patients
    assert res.sld.shape == (50, test.n_patients, len(times))
    assert not np.isnan(res.sld).any()
    # Event indicators are binary; times non-negative.
    for ep in ("pfs", "os"):
        t, e = res._event_arrays(ep)
        assert set(np.unique(e)).issubset({0, 1})
        assert (t >= 0).all()


def test_predicted_event_prob_in_unit_interval(split):
    train, test = split
    m = MarginalResamplingModel().fit(train)
    res = m.simulate(test.covariates(), n_draws=80, seed=2)
    p = res.predicted_event_prob("pfs", 180.0)
    assert p.shape == (test.n_patients,)
    assert ((p >= 0) & (p <= 1)).all()
    # Monotone: P(by 360) >= P(by 180).
    assert (res.predicted_event_prob("pfs", 360.0) + 1e-9 >= p).all()


def test_backoff_to_marginal_for_sparse_query(split):
    train, _ = split
    m = MarginalResamplingModel(min_donors=8).fit(train)
    # A query with an unusual covariate combo still simulates (coarse backoff).
    import pandas as pd

    q = pd.DataFrame(
        {"age": [95.0], "ecog_ps": [2], "stage": ["IIIB"], "histology": ["other"],
         "sex": ["F"], "smoking": ["never"], "prior_lines": [2],
         "baseline_sld_mm": [40.0], "n_target_lesions": [1]},
        index=["q1"],
    )
    res = m.simulate(q, n_draws=20, seed=3)
    assert res.n_patients == 1
    assert np.isfinite(res.pfs_time).all()
