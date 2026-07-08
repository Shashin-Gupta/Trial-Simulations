"""Data-profiling summaries for the real PDS trials (Step 1 sanity checks).

Produces, per trial: sample size, baseline covariate distributions, missingness,
follow-up duration, event maturity, and aggregate BOR — the checks that must
pass before any modelling (methodology §6). Results are returned as plain dicts /
DataFrames so a script or notebook can serialise or render them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vca._km import kaplan_meier
from vca.data_processing.pds_trials import BOR_LEVELS, RealTrial
from vca.data_processing.schema import BASELINE_REQUIRED

DAYS_PER_MONTH = 30.4375
_NUMERIC_COVS = ("age", "ecog_ps", "prior_lines", "baseline_sld_mm", "n_target_lesions")
_CATEG_COVS = ("sex", "stage", "histology", "smoking")


def _km_median(time, event) -> float:
    time = pd.to_numeric(pd.Series(time), errors="coerce").to_numpy(float)
    event = pd.to_numeric(pd.Series(event), errors="coerce").to_numpy(float)
    ok = np.isfinite(time) & np.isfinite(event)
    if ok.sum() < 5:
        return float("nan")
    t, s = kaplan_meier(time[ok], event[ok].astype(int))
    below = t[s <= 0.5]
    return float(below[0]) if below.size else float("nan")


def _endpoint_summary(ev: pd.DataFrame, ep: str) -> dict:
    t = pd.to_numeric(ev[f"{ep}_time_days"], errors="coerce")
    e = pd.to_numeric(ev[f"{ep}_event"], errors="coerce")
    n_known = int(e.notna().sum())
    if n_known == 0:
        return {"available": False, "median_months": None, "events": 0,
                "n": 0, "event_rate": None, "max_followup_months": None}
    med = _km_median(t, e)
    return {
        "available": True,
        "median_months": None if not np.isfinite(med) else round(med / DAYS_PER_MONTH, 2),
        "events": int(e.fillna(0).sum()),
        "n": n_known,
        "event_rate": round(float(e.fillna(0).sum()) / n_known, 3),
        "max_followup_months": round(float(np.nanmax(t.to_numpy(float))) / DAYS_PER_MONTH, 1),
    }


def profile_trial(rt: RealTrial) -> dict:
    """Return a structured profile dict for one trial."""
    b = rt.data.baseline
    ev = rt.data.events
    n = rt.n

    numeric_stats = {}
    for c in _NUMERIC_COVS:
        col = pd.to_numeric(b[c], errors="coerce") if c in b else pd.Series(dtype=float)
        numeric_stats[c] = {
            "mean": None if col.dropna().empty else round(float(col.mean()), 2),
            "sd": None if col.dropna().empty else round(float(col.std()), 2),
            "min": None if col.dropna().empty else round(float(col.min()), 2),
            "max": None if col.dropna().empty else round(float(col.max()), 2),
            "pct_missing": round(float(col.isna().mean()) * 100, 1),
        }
    categ_stats = {
        c: {str(k): int(v) for k, v in b[c].value_counts(dropna=False).items()}
        for c in _CATEG_COVS if c in b
    }
    missingness = {c: round(float(b[c].isna().mean()) * 100, 1)
                   for c in BASELINE_REQUIRED if c in b}

    bor_counts = rt.bor.value_counts().reindex(list(BOR_LEVELS) + ["NE"], fill_value=0)
    return {
        "trial_id": rt.trial_id,
        "n_patients": n,
        "regimen": rt.meta.get("regimen"),
        "line": rt.meta.get("line"),
        "histology_label": rt.meta.get("histology_label"),
        "reader": rt.meta.get("reader"),
        "pfs": _endpoint_summary(ev, "pfs"),
        "os": _endpoint_summary(ev, "os"),
        "pfs_convention": rt.meta.get("pfs_convention"),
        "os_convention": rt.meta.get("os_convention"),
        "numeric_covariates": numeric_stats,
        "categorical_covariates": categ_stats,
        "missingness_pct": missingness,
        "bor_counts": {k: int(v) for k, v in bor_counts.items()},
        "bor_proportions_evaluable": {k: round(float(v), 3)
                                      for k, v in rt.bor_proportions().items()},
        "n_longitudinal_rows": int(len(rt.data.longitudinal)),
    }


def profiles_to_frame(profiles: dict[str, dict]) -> pd.DataFrame:
    """One-row-per-trial headline table for quick comparison / CSV export."""
    rows = []
    for p in profiles.values():
        rows.append({
            "trial_id": p["trial_id"],
            "n": p["n_patients"],
            "regimen": p["regimen"],
            "line": p["line"],
            "histology": p["histology_label"],
            "reader": p["reader"],
            "age_mean": p["numeric_covariates"]["age"]["mean"],
            "pct_female": _pct_female(p),
            "pfs_median_mo": p["pfs"]["median_months"],
            "pfs_events": p["pfs"]["events"],
            "os_median_mo": p["os"]["median_months"],
            "os_events": p["os"]["events"],
            "os_maturity": p["os"]["event_rate"],
            "orr_pct": round(100 * (p["bor_proportions_evaluable"].get("CR", 0)
                                    + p["bor_proportions_evaluable"].get("PR", 0)), 1),
            "bor_ne_pct": round(100 * p["bor_counts"]["NE"] / max(p["n_patients"], 1), 1),
        })
    return pd.DataFrame(rows)


def _pct_female(p: dict) -> float | None:
    sex = p["categorical_covariates"].get("sex", {})
    total = sum(sex.values()) or 1
    return round(100 * sex.get("F", 0) / total, 1)
