import numpy as np

from vca.validation import metrics as M


def test_crps_normal_matches_closed_form():
    # For X ~ N(mu, sigma) and observation y, CRPS has a closed form. Check the
    # sample estimator converges to it.
    rng = np.random.default_rng(0)
    mu, sigma, y = 5.0, 2.0, 6.0
    samples = rng.normal(mu, sigma, (1, 40000))
    z = (y - mu) / sigma
    from scipy.stats import norm

    closed = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    est = M.crps(samples, np.array([y]))
    assert abs(est - closed) < 0.05


def test_crps_lower_when_centered_on_truth():
    rng = np.random.default_rng(1)
    y = np.array([10.0])
    good = rng.normal(10.0, 1.0, (1, 5000))
    bad = rng.normal(15.0, 1.0, (1, 5000))
    assert M.crps(good, y) < M.crps(bad, y)


def test_interval_coverage_is_nominal_for_correct_model():
    rng = np.random.default_rng(2)
    n = 4000
    truth = rng.normal(0, 1, n)
    samples = rng.normal(0, 1, (n, 500))  # correctly specified predictive
    cov = M.interval_coverage(samples, truth, level=0.9)
    assert abs(cov - 0.9) < 0.03


def test_brier_perfect_prediction_low():
    # No censoring; predict event-by-t perfectly -> Brier ~ 0.
    n = 500
    rng = np.random.default_rng(3)
    time = rng.uniform(50, 400, n)
    event = np.ones(n, int)
    t = 200.0
    pred = (time <= t).astype(float)  # perfect
    bs = M.brier_score_ipcw(pred, time, event, t)
    assert bs < 0.02


def test_brier_uninformative_is_higher_than_informative():
    n = 800
    rng = np.random.default_rng(4)
    time = rng.exponential(200, n)
    event = np.ones(n, int)
    t = 150.0
    truth = (time <= t).astype(float)
    good = np.clip(truth * 0.9 + 0.05, 0, 1)
    guess = np.full(n, 0.5)
    assert M.brier_score_ipcw(good, time, event, t) < M.brier_score_ipcw(guess, time, event, t)


def test_calibration_perfect_model_low_ece():
    # Labels drawn with probability equal to the prediction -> well calibrated.
    rng = np.random.default_rng(5)
    n = 5000
    p = rng.uniform(0, 1, n)
    label = (rng.uniform(0, 1, n) < p).astype(int)
    # Encode as an event-by-t problem with no censoring at t=1.
    time = np.where(label == 1, 0.5, 2.0)
    event = np.ones(n, int)
    cal = M.landmark_calibration(p, time, event, t=1.0, n_bins=10)
    assert cal.calibration_error < 0.05
    assert cal.n_used == n
