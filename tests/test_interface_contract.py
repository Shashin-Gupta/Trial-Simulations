"""Every TrajectoryModel must satisfy the same fit/simulate contract."""

import numpy as np
import pytest
from conftest import requires_numpyro

from vca.models.base import SimulationResult, TrajectoryModel
from vca.models.baseline import MarginalResamplingModel


def _assert_valid_result(res: SimulationResult, n_patients: int, n_draws: int, n_times: int):
    assert isinstance(res, SimulationResult)
    assert res.n_patients == n_patients
    assert res.n_draws == n_draws
    assert res.sld.shape == (n_draws, n_patients, n_times)
    assert np.isfinite(res.sld).all()
    for ep in ("pfs", "os"):
        t, e = res._event_arrays(ep)
        assert t.shape == (n_draws, n_patients)
        assert set(np.unique(e)).issubset({0, 1})
        assert (t >= 0).all()
    # Predictive event probability is a valid, monotone probability.
    p1 = res.predicted_event_prob("pfs", 120.0)
    p2 = res.predicted_event_prob("pfs", 360.0)
    assert ((p1 >= 0) & (p1 <= 1)).all()
    assert (p2 + 1e-9 >= p1).all()
    # One-cohort sampler + events frame.
    frame = res.to_events_frame(seed=0)
    assert len(frame) == n_patients
    assert {"pfs_time_days", "pfs_event", "os_time_days", "os_event"}.issubset(frame.columns)


def _make_model(kind):
    if kind == "baseline":
        return MarginalResamplingModel(min_donors=8)
    if kind == "bayes":
        from vca.models.tgi_survival import TGISurvivalModel

        return TGISurvivalModel(num_warmup=60, num_samples=60, num_chains=1, seed=0)
    raise ValueError(kind)


def test_baseline_contract(split):
    train, test = split
    model = _make_model("baseline")
    assert isinstance(model, TrajectoryModel)
    model.fit(train)
    times = np.arange(0.0, 361.0, 30.0)
    res = model.simulate(test.covariates(), n_draws=40, times=times, seed=1)
    _assert_valid_result(res, test.n_patients, 40, len(times))


@requires_numpyro
@pytest.mark.slow
@pytest.mark.bayes
def test_bayesian_contract(split):
    train, test = split
    model = _make_model("bayes")
    assert isinstance(model, TrajectoryModel)
    model.fit(train)
    times = np.arange(0.0, 361.0, 30.0)
    res = model.simulate(test.covariates(), n_draws=40, times=times, seed=1)
    _assert_valid_result(res, test.n_patients, 40, len(times))


def test_missing_required_covariate_raises(split):
    train, test = split
    from vca.models.tgi_survival import TGISurvivalModel

    # required_covariates is declared even without fitting the sampler.
    model = TGISurvivalModel()
    model.is_fitted = True  # bypass fit to test the covariate guard directly
    bad = test.covariates().drop(columns=["baseline_sld_mm"])
    with pytest.raises(ValueError):
        model.simulate(bad, n_draws=5)
