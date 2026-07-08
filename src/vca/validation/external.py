"""External *aggregate* validation against BOR/survival-only trials.

The model is trained on _438 (the only trial with lesion trajectories). For each
of the four external trials we build a **matched synthetic population** — the
trial's real baseline covariates, with a tumour-burden anchor (baseline SLD /
target-lesion count) drawn from _438 because those trials do not record it — run
the fitted model's ``.simulate()``, roll the synthetic cohort up to aggregate
best-overall-response (BOR) proportions and PFS/OS Kaplan–Meier curves, and test
those against the trial's *real* aggregates:

* **BOR** — a χ² test (Fisher's exact when any expected cell < 5) on the
  evaluable CR/PR/SD/PD contingency table.
* **PFS / OS** — a log-rank test of the simulated vs real KM curves, with the
  simulated latent times subjected to the real cohort's empirical censoring so
  the comparison is fair (reusing :func:`vca.validation.survival.compare_survival`).

A *large* BOR p-value and a *large* log-rank p-value mean the synthetic control
arm is not detectably different from the real one — the desirable outcome. This
is deliberately a per-trial analysis (never pooled), because the scientific
question is exactly where generalisation holds and where it breaks (squamous,
second-line, different regimens).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from vca.data_processing.pds_trials import BOR_LEVELS, RealTrial
from vca.data_processing.schema import TrialData
from vca.models.base import SimulationResult, TrajectoryModel
from vca.validation.survival import SurvivalComparison, compare_survival

DAYS_PER_MONTH = 30.4375


# --------------------------------------------------------------------------- #
# RECIST best-overall-response from a simulated SLD trajectory
# --------------------------------------------------------------------------- #

_BOR_RANK = {"CR": 0, "PR": 1, "SD": 2, "PD": 3}


def classify_bor(
    times: np.ndarray,
    sld: np.ndarray,
    *,
    cr_abs_mm: float = 5.0,
    pr_frac: float = -0.30,
    pd_frac: float = 0.20,
    pd_abs_mm: float = 5.0,
    horizon_days: float | None = 540.0,
) -> str:
    """RECIST 1.1-style best overall response from one SLD trajectory.

    The RECIST best-overall-response algorithm is followed on the *target-lesion*
    sum: post-baseline visits are walked in time; assessment stops at the first
    progression, and BOR is the best-ranked visit response (CR > PR > SD > PD)
    seen up to that point. So a patient who is stable and later progresses is
    SD (their best), while a patient who progresses at the first post-baseline
    visit is PD.

    Per-visit response (vs baseline, and vs the running nadir for progression):

    * **CR** — SLD at (near) zero (``<= cr_abs_mm``).
    * **PR** — ``>= 30%`` below baseline.
    * **PD** — ``>= 20%`` above the running nadir *and* ``>= 5 mm`` absolute.
    * **SD** — anything else.

    Simplifications vs a full RECIST read (documented in methodology): no
    confirmation requirement, no minimum SD interval, and non-target lesions /
    new lesions are not modelled (only the target-lesion SLD trajectory is).
    ``horizon_days`` bounds the assessment window (default 18 months).
    """
    times = np.asarray(times, float)
    sld = np.asarray(sld, float)
    if horizon_days is not None:
        keep = times <= horizon_days
        times, sld = times[keep], sld[keep]
    finite = np.isfinite(sld)
    times, sld = times[finite], sld[finite]
    if sld.size == 0 or not np.isfinite(sld[0]) or sld[0] <= 0:
        return "NE"

    base = sld[0]
    nadir = base
    seen: list[str] = []
    for s in sld[1:]:  # baseline itself is not an assessment
        rise = s - nadir
        if rise >= pd_abs_mm and rise / max(nadir, 1e-6) >= pd_frac:
            seen.append("PD")
            break  # progression ends the assessment period
        if s <= cr_abs_mm:
            seen.append("CR")
        elif (s - base) / base <= pr_frac:
            seen.append("PR")
        else:
            seen.append("SD")
        nadir = min(nadir, s)
    if not seen:
        return "SD"
    return min(seen, key=lambda r: _BOR_RANK[r])


def simulated_bor_counts(
    result: SimulationResult, *, seed: int = 0, **bor_kwargs
) -> pd.Series:
    """One BOR label per synthetic patient (a single trajectory draw each).

    Classification uses the noise-free expected trajectory (``sld_mean``) when
    available, because RECIST categories describe the *underlying* tumour
    response; applying the thresholds to single noisy scans would spuriously
    reclassify stable patients as responders/progressors.
    """
    rng = np.random.default_rng(seed)
    traj = result.sld_mean if result.sld_mean is not None else result.sld
    n_draws, n_pat, _ = traj.shape
    picks = rng.integers(0, n_draws, size=n_pat)
    labels = [classify_bor(result.times, traj[picks[j], j, :], **bor_kwargs)
              for j in range(n_pat)]
    counts = pd.Series(labels).value_counts().reindex(list(BOR_LEVELS), fill_value=0)
    return counts


# --------------------------------------------------------------------------- #
# Matched synthetic population
# --------------------------------------------------------------------------- #

def matched_population(
    real: RealTrial, donor: TrialData, *, n: int | None = None, seed: int = 0
) -> pd.DataFrame:
    """Covariate frame for a synthetic cohort matched to ``real``.

    Uses the trial's real baseline covariates (age, sex, stage, histology,
    smoking, ECOG, prior lines) and attaches a tumour-burden anchor
    (``baseline_sld_mm``, ``n_target_lesions``) resampled from the ``donor``
    (_438) measurable-disease cohort, because the external trials do not record
    it. Sampling the (SLD, n_target) pair jointly preserves their association.
    """
    rng = np.random.default_rng(seed)
    base = real.data.baseline.copy()
    if n is not None and n != len(base):
        base = base.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    dsld = pd.to_numeric(donor.baseline["baseline_sld_mm"], errors="coerce")
    dnt = pd.to_numeric(donor.baseline["n_target_lesions"], errors="coerce")
    pool = pd.DataFrame({"sld": dsld, "nt": dnt}).dropna().to_numpy()
    idx = rng.integers(0, len(pool), size=len(base))
    base["baseline_sld_mm"] = pool[idx, 0]
    base["n_target_lesions"] = pool[idx, 1]
    base.index = base["patient_id"].astype(str)
    return base


# --------------------------------------------------------------------------- #
# per-trial external validation
# --------------------------------------------------------------------------- #

@dataclass
class ExternalTrialResult:
    trial_id: str
    n_real: int
    n_sim: int
    regimen: str
    line: str
    histology_label: str
    bor: dict = field(default_factory=dict)
    pfs: dict | None = None
    os: dict | None = None
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _bor_test(real_counts: pd.Series, sim_counts: pd.Series) -> dict:
    """χ² (or Fisher for 2×2 / sparse) on the evaluable CR/PR/SD/PD table."""
    table = np.vstack([real_counts.to_numpy(float), sim_counts.to_numpy(float)])
    # drop all-zero columns (a category absent in both arms carries no signal)
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    labels = [lab for lab, k in zip(BOR_LEVELS, keep, strict=False) if k]
    result = {"categories": labels,
              "real": real_counts.to_numpy(int).tolist(),
              "sim": sim_counts.to_numpy(int).tolist()}
    if table.shape[1] < 2 or table.sum() == 0:
        result.update(test="none", statistic=None, p_value=None)
        return result
    chi2, p, dof, expected = stats.chi2_contingency(table)
    if (expected < 5).any() and table.shape[1] == 2:
        _, p = stats.fisher_exact(table)
        result.update(test="fisher_exact", statistic=None, p_value=float(p))
    else:
        result.update(test="chi2", statistic=float(chi2), dof=int(dof),
                      min_expected=float(expected.min()), p_value=float(p))
    return result


def _surv_dict(cmp: SurvivalComparison) -> dict:
    d = cmp.to_dict()
    d["real_median_months"] = (None if d["real_median_days"] is None
                               else round(d["real_median_days"] / DAYS_PER_MONTH, 2))
    d["sim_median_months"] = (None if d["sim_median_days"] is None
                              else round(d["sim_median_days"] / DAYS_PER_MONTH, 2))
    return d


def external_validate_trial(
    model: TrajectoryModel,
    real: RealTrial,
    donor: TrialData,
    *,
    n_draws: int = 300,
    seed: int = 0,
    horizon_days: float | None = 540.0,
) -> tuple[ExternalTrialResult, SimulationResult, dict]:
    """Run the full external aggregate validation for one trial.

    Returns the structured result, the underlying :class:`SimulationResult`, and
    a dict of the fitted survival comparisons (for plotting).
    """
    cov = matched_population(real, donor, seed=seed)
    times = np.arange(0.0, 721.0, 30.0)
    result = model.simulate(cov, n_draws=n_draws, times=times, seed=seed)

    # --- BOR ----------------------------------------------------------------
    real_bor = real.bor[real.bor.isin(BOR_LEVELS)]
    real_counts = real_bor.value_counts().reindex(list(BOR_LEVELS), fill_value=0)
    sim_counts = simulated_bor_counts(result, seed=seed + 3, horizon_days=horizon_days)
    bor = _bor_test(real_counts, sim_counts)
    bor["real_proportions"] = {k: round(float(v), 3) for k, v in
                               (real_counts / real_counts.sum()).items()}
    bor["sim_proportions"] = {k: round(float(v), 3) for k, v in
                              (sim_counts / sim_counts.sum()).items()}

    # --- survival -----------------------------------------------------------
    ev = real.data.events
    surv_cmps = {}
    pfs_d = os_d = None
    for ep in ("pfs", "os"):
        rt_ = pd.to_numeric(ev[f"{ep}_time_days"], errors="coerce").to_numpy(float)
        re_ = pd.to_numeric(ev[f"{ep}_event"], errors="coerce")
        if re_.notna().sum() == 0 or np.nansum(re_.to_numpy(float)) == 0:
            continue  # endpoint not available for this trial (e.g. 133 PFS)
        re_ = re_.to_numpy(float)
        ok = np.isfinite(rt_) & np.isfinite(re_)
        sim_latent, _ = result.sample_one_per_patient(ep, seed=seed + 7)
        # align sim cohort size to the real endpoint's evaluable patients
        m = min(ok.sum(), sim_latent.size)
        cmp = compare_survival(ep, sim_latent[:m], rt_[ok][:m], re_[ok][:m].astype(int),
                               seed=seed + 11)
        surv_cmps[ep] = cmp
        if ep == "pfs":
            pfs_d = _surv_dict(cmp)
        else:
            os_d = _surv_dict(cmp)

    res = ExternalTrialResult(
        trial_id=real.trial_id, n_real=real.n, n_sim=len(cov),
        regimen=real.meta.get("regimen", ""), line=real.meta.get("line", ""),
        histology_label=real.meta.get("histology_label", ""),
        bor=bor, pfs=pfs_d, os=os_d,
    )
    return res, result, surv_cmps
