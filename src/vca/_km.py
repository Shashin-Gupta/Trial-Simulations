"""Minimal, dependency-free Kaplan-Meier utilities (numpy only).

Used by the baseline model to sample *latent* event times from an estimated
survival curve, and by tests. The heavier survival machinery in
``vca.validation.survival`` uses ``lifelines`` for curves, confidence bands, and
the log-rank test; this module deliberately avoids that dependency so the core
sampling path is tiny and easy to reason about.
"""

from __future__ import annotations

import numpy as np


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival estimate.

    Parameters
    ----------
    time
        Observed times (event or censoring).
    event
        Event indicator, ``1`` = event, ``0`` = right-censored.

    Returns
    -------
    (event_times, survival)
        ``event_times`` are the distinct times at which the estimate steps down
        (event times only); ``survival`` is S(t) just after each step. Both are
        1-D arrays of equal length. If there are no events, returns empty arrays.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]

    distinct = np.unique(time[event == 1])
    if distinct.size == 0:
        return np.array([]), np.array([])

    surv = np.empty(distinct.size)
    s = 1.0
    for i, t in enumerate(distinct):
        at_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        surv[i] = s
    return distinct, surv


def sample_from_km(
    event_times: np.ndarray,
    survival: np.ndarray,
    n: int,
    rng: np.random.Generator,
    max_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n`` latent event times by inverting a KM survival curve.

    A draw ``u ~ Uniform(0, 1)`` is mapped to the first event time whose
    survival drops to or below ``u``. Probability mass below the curve's final
    plateau ``S_min`` (the part of the population that outlives all observed
    events) is treated as *censored at* ``max_time`` — i.e. the event is known
    only to occur beyond the observed follow-up.

    Returns ``(times, events)`` where ``events`` is 1 for realised event times
    and 0 for the beyond-horizon plateau mass.
    """
    if event_times.size == 0:
        # No events observed: everyone is censored at the horizon.
        return np.full(n, max_time), np.zeros(n, dtype=int)

    u = rng.random(n)
    s_min = survival[-1]
    times = np.empty(n)
    events = np.ones(n, dtype=int)

    beyond = u < s_min
    times[beyond] = max_time
    events[beyond] = 0

    # For u >= s_min, invert: first time with survival <= u.
    idx = np.searchsorted(-survival, -u[~beyond], side="left")
    idx = np.clip(idx, 0, event_times.size - 1)
    times[~beyond] = event_times[idx]
    return times, events
