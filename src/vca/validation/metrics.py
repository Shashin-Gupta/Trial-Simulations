"""Validation metrics: calibration, interval coverage, Brier (IPCW), CRPS.

These operate on plain arrays so they are easy to unit-test and reuse. The
orchestration in :mod:`vca.validation.pipeline` wires them to a fitted model and
a held-out :class:`~vca.data_processing.schema.TrialData`.

Censoring is handled honestly:

- **IPCW Brier score** (Graf et al., 1999) reweights by the Kaplan-Meier estimate
  of the *censoring* distribution, giving an unbiased time-dependent Brier score
  under independent censoring.
- **Landmark calibration** uses the complete-case-at-landmark construction
  (patients whose event status at the landmark is known), which is simple and
  transparent; its bias under informative censoring is noted in the docs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vca._km import kaplan_meier

# --------------------------------------------------------------------------- #
# Censoring distribution
# --------------------------------------------------------------------------- #

def _censoring_km(time: np.ndarray, event: np.ndarray):
    """KM estimate of the censoring survival function G(u) = P(C > u)."""
    # The censoring "event" is 1 - event.
    return kaplan_meier(time, 1 - np.asarray(event))


def _step_lookup(event_times: np.ndarray, survival: np.ndarray, u, floor: float = 1e-8):
    """Right-continuous step-function lookup S(u) with a small floor.

    Returns S just after the last step at or before ``u`` (S(0)=1 before any
    step). Floored away from zero so it is safe as an IPCW denominator.
    """
    u = np.atleast_1d(np.asarray(u, float))
    if event_times.size == 0:
        return np.ones_like(u)
    idx = np.searchsorted(event_times, u, side="right") - 1
    out = np.where(idx < 0, 1.0, survival[np.clip(idx, 0, survival.size - 1)])
    return np.maximum(out, floor)


# --------------------------------------------------------------------------- #
# Time-dependent Brier score with IPCW
# --------------------------------------------------------------------------- #

def brier_score_ipcw(
    pred_event_prob: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t: float,
) -> float:
    """IPCW time-dependent Brier score at horizon ``t`` (lower is better).

    Parameters
    ----------
    pred_event_prob
        Predicted P(event by ``t``) for each patient, in [0, 1].
    time, event
        Observed times and event indicators (1 = event, 0 = censored).
    t
        Landmark horizon (same units as ``time``).

    Notes
    -----
    Predicting event-by-``t`` is scored against survival ``S = 1 - p``:
    patients with an event by ``t`` contribute ``S^2`` weighted by ``1/G(t_i^-)``;
    patients known event-free at ``t`` contribute ``p^2`` weighted by ``1/G(t)``;
    patients censored before ``t`` are uninformative (weight 0).
    """
    pred_event_prob = np.asarray(pred_event_prob, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    S = 1.0 - pred_event_prob

    g_times, g_surv = _censoring_km(time, event)
    G_t = _step_lookup(g_times, g_surv, t)[0]
    G_ti = _step_lookup(g_times, g_surv, np.clip(time - 1e-9, 0, None))

    had_event = (time <= t) & (event == 1)
    survived = time > t

    contrib = np.zeros_like(pred_event_prob)
    contrib[had_event] = (S[had_event] ** 2) / G_ti[had_event]
    contrib[survived] = (pred_event_prob[survived] ** 2) / G_t
    # censored-before-t stay 0
    return float(np.mean(contrib))


# --------------------------------------------------------------------------- #
# Landmark calibration (binary event-by-t)
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationCurve:
    bin_pred: np.ndarray      # mean predicted probability per bin
    bin_obs: np.ndarray       # observed event frequency per bin
    bin_count: np.ndarray     # patients per bin
    n_used: int               # patients with known status at landmark
    calibration_error: float  # weighted mean |obs - pred| (ECE-style)


def landmark_calibration(
    pred_event_prob: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t: float,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Calibration of predicted event-by-``t`` against observed frequency.

    Uses complete-case-at-landmark: keep patients with an event by ``t`` (label
    1) or known event-free at ``t`` (followed past ``t``, label 0); drop patients
    censored before ``t`` (status unknown). Bins by predicted probability.
    """
    pred = np.asarray(pred_event_prob, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)

    had_event = (time <= t) & (event == 1)
    survived = time > t
    keep = had_event | survived
    p = pred[keep]
    y = had_event[keep].astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bin_pred = np.full(n_bins, np.nan)
    bin_obs = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        bin_count[b] = int(m.sum())
        if m.any():
            bin_pred[b] = p[m].mean()
            bin_obs[b] = y[m].mean()

    valid = bin_count > 0
    if valid.any():
        w = bin_count[valid] / bin_count[valid].sum()
        ece = float(np.sum(w * np.abs(bin_obs[valid] - bin_pred[valid])))
    else:
        ece = float("nan")
    return CalibrationCurve(bin_pred, bin_obs, bin_count, int(keep.sum()), ece)


# --------------------------------------------------------------------------- #
# Prediction-interval coverage
# --------------------------------------------------------------------------- #

def interval_coverage(
    samples: np.ndarray, observed: np.ndarray, level: float = 0.9
) -> float:
    """Empirical coverage of central ``level`` predictive intervals.

    ``samples`` is (n_obs, n_draws); ``observed`` is (n_obs,). Returns the
    fraction of observations inside their per-observation central interval.
    """
    samples = np.asarray(samples, float)
    observed = np.asarray(observed, float)
    lo = np.nanquantile(samples, (1 - level) / 2, axis=1)
    hi = np.nanquantile(samples, 1 - (1 - level) / 2, axis=1)
    inside = (observed >= lo) & (observed <= hi)
    return float(np.mean(inside))


def coverage_table(
    samples: np.ndarray, observed: np.ndarray, levels=(0.5, 0.8, 0.9, 0.95)
) -> dict[float, float]:
    """Coverage at several nominal levels (calibration of the intervals)."""
    return {lev: interval_coverage(samples, observed, lev) for lev in levels}


# --------------------------------------------------------------------------- #
# Continuous ranked probability score (CRPS)
# --------------------------------------------------------------------------- #

def crps_samples(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Per-observation CRPS from an ensemble forecast (lower is better).

    ``samples`` is (n_obs, n_draws); ``observed`` is (n_obs,). Uses the
    sample-based estimator
    ``CRPS = E|X - y| - 0.5 E|X - X'|`` with the O(m log m) sorted form for the
    second term.
    """
    samples = np.asarray(samples, float)
    observed = np.asarray(observed, float)
    n_obs, m = samples.shape
    xs = np.sort(samples, axis=1)

    term1 = np.mean(np.abs(xs - observed[:, None]), axis=1)
    # E|X - X'| = (2 / m^2) * sum_i (2i - m - 1) * x_(i),  i = 1..m
    i = np.arange(1, m + 1)
    weights = (2 * i - m - 1)
    term2 = (2.0 / m**2) * np.sum(weights[None, :] * xs, axis=1)
    return term1 - 0.5 * term2


def crps(samples: np.ndarray, observed: np.ndarray) -> float:
    """Mean CRPS over observations."""
    return float(np.mean(crps_samples(samples, observed)))
