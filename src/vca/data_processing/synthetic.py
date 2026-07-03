"""Synthetic advanced-NSCLC trial data conforming to the canonical schema.

This module generates realistic *fake* patient-level data with a known
data-generating process. It exists for three reasons:

1. **Runnable pipeline without gated data.** Project Data Sphere / SEER data are
   access-gated and cannot be committed. The synthetic generator lets anyone run
   the full data -> model -> validation pipeline end to end immediately.
2. **Ground-truth validation.** Because we know the true tumour-growth and
   survival parameters, we can check that a model *recovers* them and that the
   validation metrics behave as expected on a case with a known answer.
3. **Unit tests.** Deterministic, fast, dependency-light fixtures.

Data-generating process
------------------------
Tumour size follows a Stein bi-exponential tumour-growth-inhibition (TGI) model

    SLD_i(t) = SLD0_i * ( exp(-d_i * t) + exp(g_i * t) - 1 )

where ``d_i`` is a per-patient shrinkage rate and ``g_i`` a regrowth rate, both
depending on baseline covariates plus a random effect. Time-to-progression and
time-to-death are Weibull, with the hazard increasing in the *same* latent
growth rate ``g_i`` — so tumour dynamics and survival are genuinely coupled,
exactly the structure the joint TGI-survival model is meant to exploit.

This is a *plausible* generator, not a calibrated one; it is not a substitute
for validation against real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vca.data_processing.schema import TrialData

VISIT_INTERVAL_DAYS = 42  # RECIST reassessment roughly every 6 weeks
DEFAULT_HORIZON_DAYS = 720


@dataclass
class SyntheticTruth:
    """Latent per-patient parameters of the generator (for ground-truth checks)."""

    patient_id: np.ndarray
    sld0: np.ndarray
    shrink_rate: np.ndarray  # d_i, per day
    growth_rate: np.ndarray  # g_i, per day


def _draw_categorical(rng, levels, probs, n):
    return rng.choice(levels, size=n, p=probs)


def tgi_trajectory(sld0, shrink, growth, t):
    """Stein bi-exponential SLD as a function of time (days).

    ``sld0``, ``shrink``, ``growth`` may be scalars or arrays broadcastable
    against ``t``. Result is clipped at a small positive floor.
    """
    y = sld0 * (np.exp(-shrink * t) + np.exp(growth * t) - 1.0)
    return np.maximum(y, 1e-3)


def make_synthetic_nsclc(
    n_patients: int = 400,
    *,
    seed: int = 0,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    study_id: str = "SYNTH-NSCLC-1",
    return_truth: bool = False,
):
    """Generate a synthetic advanced-NSCLC comparator arm.

    Parameters
    ----------
    n_patients
        Number of virtual patients.
    seed
        RNG seed for reproducibility.
    horizon_days
        Maximum follow-up; administrative censoring is drawn below this.
    return_truth
        If ``True``, also return a :class:`SyntheticTruth` with latent params.

    Returns
    -------
    TrialData  (and optionally SyntheticTruth)
    """
    rng = np.random.default_rng(seed)
    n = n_patients
    pid = np.array([f"S{ i:05d}" for i in range(n)])

    # -- baseline covariates -------------------------------------------------
    age = np.clip(rng.normal(64, 9, n), 34, 88)
    sex = _draw_categorical(rng, ["M", "F"], [0.58, 0.42], n)
    ecog = _draw_categorical(rng, [0, 1, 2], [0.40, 0.50, 0.10], n)
    stage = _draw_categorical(rng, ["IIIB", "IV"], [0.25, 0.75], n)
    histology = _draw_categorical(rng, ["non_squamous", "squamous", "other"], [0.65, 0.30, 0.05], n)
    smoking = _draw_categorical(
        rng, ["current", "former", "never", "unknown"], [0.25, 0.53, 0.15, 0.07], n
    )
    prior_lines = _draw_categorical(rng, [0, 1, 2], [0.50, 0.35, 0.15], n)
    n_target = rng.integers(1, 6, n)
    # Baseline SLD (mm): lognormal, mildly increasing with number of lesions.
    baseline_sld = np.round(
        np.exp(rng.normal(np.log(55) + 0.12 * n_target, 0.45)), 1
    ).clip(10, 350)
    # PD-L1 frequently untested in older comparator arms -> mostly missing.
    pdl1 = np.where(rng.random(n) < 0.6, np.nan, rng.integers(0, 101, n).astype(float))
    egfr = _draw_categorical(
        rng, ["unknown", "wildtype", "mutant"], [0.55, 0.35, 0.10], n
    )
    alk = _draw_categorical(rng, ["unknown", "negative", "positive"], [0.6, 0.37, 0.03], n)

    # -- covariate-driven latent tumour dynamics -----------------------------
    # Standardised covariate contributions to log-rates. Poorer prognosis
    # (higher ECOG, squamous histology, more prior lines) -> less shrinkage and
    # faster regrowth.
    ecog_z = (ecog - 0.7)
    squamous = (histology == "squamous").astype(float)
    lines_z = (prior_lines - 0.65)

    log_shrink = (
        np.log(0.0045)
        - 0.20 * ecog_z
        - 0.15 * squamous
        - 0.12 * lines_z
        + rng.normal(0, 0.35, n)
    )
    log_growth = (
        np.log(0.0018)
        + 0.30 * ecog_z
        + 0.18 * squamous
        + 0.15 * lines_z
        + rng.normal(0, 0.40, n)
    )
    shrink = np.exp(log_shrink)
    growth = np.exp(log_growth)
    sld0 = baseline_sld * np.exp(rng.normal(0, 0.03, n))  # near-exact anchor to baseline

    # -- survival, coupled to growth rate ------------------------------------
    # Weibull time-to-progression and time-to-death; log-scale decreases (hazard
    # increases) with the latent growth rate and with ECOG.
    growth_z = (log_growth - np.log(0.0018))
    lp_prog = 5.55 - 0.55 * growth_z - 0.25 * ecog_z - 0.10 * lines_z
    lp_death = 6.35 - 0.45 * growth_z - 0.35 * ecog_z - 0.10 * lines_z
    k_prog, k_death = 1.35, 1.25
    prog_latent = np.exp(lp_prog) * rng.weibull(k_prog, n)
    death_latent = np.exp(lp_death) * rng.weibull(k_death, n)
    pfs_raw = np.minimum(prog_latent, death_latent)  # progression OR death

    # Administrative censoring from staggered accrual + fixed data cutoff.
    censor = rng.uniform(0.45 * horizon_days, horizon_days, n)

    os_time = np.minimum(death_latent, censor)
    os_event = (death_latent <= censor).astype(int)
    pfs_time = np.minimum(pfs_raw, censor)
    pfs_event = (pfs_raw <= censor).astype(int)

    baseline = pd.DataFrame(
        {
            "patient_id": pid,
            "study_id": study_id,
            "treatment": "synthetic_comparator",
            "age": np.round(age, 1),
            "sex": sex,
            "ecog_ps": ecog,
            "stage": stage,
            "histology": histology,
            "smoking": smoking,
            "prior_lines": prior_lines,
            "baseline_sld_mm": baseline_sld,
            "n_target_lesions": n_target,
            "egfr_status": egfr,
            "alk_status": alk,
            "pdl1_tps": pdl1,
        }
    )

    events = pd.DataFrame(
        {
            "patient_id": pid,
            "pfs_time_days": np.round(pfs_time, 1),
            "pfs_event": pfs_event,
            "os_time_days": np.round(os_time, 1),
            "os_event": os_event,
        }
    )

    # -- longitudinal SLD measurements ---------------------------------------
    # Measure at scheduled visits until the patient leaves follow-up (documented
    # progression or death), with one post-baseline confirmation visit allowed.
    rows = []
    for i in range(n):
        last_visit = min(pfs_time[i] + VISIT_INTERVAL_DAYS, os_time[i], horizon_days)
        visit_times = np.arange(0, last_visit + 1, VISIT_INTERVAL_DAYS)
        true_sld = tgi_trajectory(sld0[i], shrink[i], growth[i], visit_times)
        # Multiplicative lognormal measurement error (~7% CV).
        obs_sld = true_sld * np.exp(rng.normal(0, 0.07, size=visit_times.shape))
        for t, y in zip(visit_times, obs_sld):
            rows.append((pid[i], float(round(t, 1)), float(round(y, 1))))
    longitudinal = pd.DataFrame(rows, columns=["patient_id", "time_days", "sld_mm"])

    td = TrialData(baseline=baseline, longitudinal=longitudinal, events=events).validate()

    if return_truth:
        truth = SyntheticTruth(
            patient_id=pid, sld0=sld0, shrink_rate=shrink, growth_rate=growth
        )
        return td, truth
    return td


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    td = make_synthetic_nsclc(200, seed=1)
    print(td)
    print(td.baseline.head())
    print(
        "median PFS (days):",
        td.events.loc[td.events.pfs_event == 1, "pfs_time_days"].median(),
    )
