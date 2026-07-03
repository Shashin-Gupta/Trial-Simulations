"""Survival-curve comparison: Kaplan-Meier, log-rank, median survival.

Compares a model's simulated survival to real held-out patients. To keep the
comparison fair, simulated *latent* event times are subjected to the test set's
empirical censoring pattern before estimating the simulated KM curve — otherwise
an uncensored simulated curve would be compared against a censored real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


@dataclass
class KMEstimate:
    timeline: np.ndarray
    survival: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    median: float


def km_estimate(time: np.ndarray, event: np.ndarray, label: str = "") -> KMEstimate:
    kmf = KaplanMeierFitter(label=label or "KM")
    kmf.fit(np.asarray(time, float), np.asarray(event, int))
    ci = kmf.confidence_interval_
    return KMEstimate(
        timeline=kmf.timeline,
        survival=kmf.survival_function_.iloc[:, 0].to_numpy(),
        ci_lower=ci.iloc[:, 0].to_numpy(),
        ci_upper=ci.iloc[:, 1].to_numpy(),
        median=float(kmf.median_survival_time_),
    )


def apply_empirical_censoring(
    latent_time: np.ndarray,
    censoring_pool: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Censor simulated latent times using resampled real follow-up times.

    Each simulated patient is assigned a censoring time drawn with replacement
    from ``censoring_pool`` (the real cohort's times under observation). The
    observed simulated time is ``min(latent, C)`` with event indicator
    ``latent <= C``. This matches the marginal censoring of the real data so the
    two KM curves and the log-rank test compare like with like.
    """
    latent_time = np.asarray(latent_time, float)
    c = rng.choice(np.asarray(censoring_pool, float), size=latent_time.shape[0], replace=True)
    obs_time = np.minimum(latent_time, c)
    event = (latent_time <= c).astype(int)
    return obs_time, event


@dataclass
class SurvivalComparison:
    endpoint: str
    real_median: float
    sim_median: float
    median_abs_diff_days: float
    logrank_stat: float
    logrank_p: float
    real_km: KMEstimate = field(repr=False)
    sim_km: KMEstimate = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "real_median_days": _nan_to_none(self.real_median),
            "sim_median_days": _nan_to_none(self.sim_median),
            "median_abs_diff_days": _nan_to_none(self.median_abs_diff_days),
            "logrank_stat": _nan_to_none(self.logrank_stat),
            "logrank_p": _nan_to_none(self.logrank_p),
        }


def _nan_to_none(x):
    x = float(x)
    return None if not np.isfinite(x) else x


def compare_survival(
    endpoint: str,
    sim_latent_time: np.ndarray,
    real_time: np.ndarray,
    real_event: np.ndarray,
    *,
    seed: int = 0,
) -> SurvivalComparison:
    """Compare a simulated cohort to real held-out patients for one endpoint.

    ``sim_latent_time`` are uncensored simulated event times (one per real
    patient); they are censoring-matched to the real cohort before comparison.
    A large log-rank p-value means the curves are *not* detectably different —
    which, for a virtual control arm, is the desirable outcome.
    """
    rng = np.random.default_rng(seed)
    real_time = np.asarray(real_time, float)
    real_event = np.asarray(real_event, int)

    sim_time, sim_event = apply_empirical_censoring(sim_latent_time, real_time, rng)

    real_km = km_estimate(real_time, real_event, label="real")
    sim_km = km_estimate(sim_time, sim_event, label="simulated")
    lr = logrank_test(real_time, sim_time, event_observed_A=real_event, event_observed_B=sim_event)

    med_diff = abs(real_km.median - sim_km.median)
    return SurvivalComparison(
        endpoint=endpoint,
        real_median=real_km.median,
        sim_median=sim_km.median,
        median_abs_diff_days=med_diff,
        logrank_stat=float(lr.test_statistic),
        logrank_p=float(lr.p_value),
        real_km=real_km,
        sim_km=sim_km,
    )
