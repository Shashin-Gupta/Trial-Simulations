import numpy as np

from vca._km import kaplan_meier, sample_from_km


def test_km_no_censoring_matches_empirical():
    # With no censoring, KM survival = 1 - empirical CDF at event times.
    time = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    event = np.ones_like(time, dtype=int)
    t, s = kaplan_meier(time, event)
    assert np.allclose(t, [1, 2, 3, 4])
    # After t=1: 4/5; after t=2 (2 events): 4/5 * 2/4 = 2/5; after 3: 2/5*1/2=1/5; after 4: 0
    assert np.allclose(s, [0.8, 0.4, 0.2, 0.0])


def test_km_with_censoring():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])  # middle censored
    t, s = kaplan_meier(time, event)
    # t=1: 2/3; censor at 2 removes one at risk; t=3: at risk=1, event -> 2/3*0=0
    assert np.allclose(t, [1.0, 3.0])
    assert np.allclose(s, [2 / 3, 0.0])


def test_sample_from_km_recovers_distribution():
    rng = np.random.default_rng(0)
    # Exponential-ish sample.
    true = rng.exponential(200.0, 4000)
    t, s = kaplan_meier(true, np.ones_like(true, dtype=int))
    draws, ev = sample_from_km(t, s, 20000, rng, max_time=1e6)
    # Median of resampled draws close to median of the source.
    assert abs(np.median(draws) - np.median(true)) / np.median(true) < 0.1


def test_sample_from_km_all_censored():
    rng = np.random.default_rng(0)
    t, s = kaplan_meier(np.array([1.0, 2.0]), np.array([0, 0]))
    draws, ev = sample_from_km(t, s, 100, rng, max_time=999.0)
    assert (draws == 999.0).all()
    assert (ev == 0).all()
