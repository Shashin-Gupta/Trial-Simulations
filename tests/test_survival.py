import numpy as np

from vca.validation.survival import compare_survival, km_estimate


def test_km_median_recovers_exponential():
    rng = np.random.default_rng(0)
    t = rng.exponential(200.0, 3000)
    km = km_estimate(t, np.ones_like(t, dtype=int))
    # Exponential median = 200 * ln 2 ~ 138.6.
    assert abs(km.median - 200 * np.log(2)) < 15


def test_identical_cohorts_not_significant():
    rng = np.random.default_rng(1)
    latent = rng.exponential(200.0, 1500)
    real_time = rng.exponential(200.0, 1500)
    real_event = np.ones_like(real_time, dtype=int)
    cmp = compare_survival("pfs", latent, real_time, real_event, seed=2)
    assert cmp.logrank_p > 0.05  # same distribution -> not distinguishable


def test_shifted_cohorts_are_detected():
    rng = np.random.default_rng(3)
    latent = rng.exponential(400.0, 1500)      # simulated much longer
    real_time = rng.exponential(150.0, 1500)   # real much shorter
    real_event = np.ones_like(real_time, dtype=int)
    cmp = compare_survival("os", latent, real_time, real_event, seed=4)
    assert cmp.logrank_p < 0.01
    assert cmp.median_abs_diff_days > 50


def test_comparison_serialises():
    rng = np.random.default_rng(5)
    cmp = compare_survival(
        "pfs", rng.exponential(200, 300),
        rng.exponential(200, 300), np.ones(300, int), seed=6,
    )
    d = cmp.to_dict()
    assert set(d) >= {"endpoint", "real_median_days", "sim_median_days", "logrank_p"}
