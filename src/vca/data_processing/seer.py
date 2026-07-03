"""SEER real-world survival data for *external calibration* (not model fitting).

SEER (NCI's Surveillance, Epidemiology, and End Results program) provides
population-based cancer incidence and survival. We use it only as an external
reference: does the simulator's baseline survival sit in the neighbourhood of
real-world lung-cancer survival for a comparable stage distribution? It is
**not** a source of RECIST trajectories and is **not** used to fit the model.

Important nuance about access
-----------------------------
SEER research microdata is obtained through the **SEER*Stat** application after a
data use agreement (approved within ~2 business days). The ``api.seer.cancer.gov``
REST service is for registry/coding integration, **not** for pulling research
microdata. The intended workflow is therefore:

1. In SEER*Stat, build a *case listing* (or rate) session for the lung/bronchus
   cohort you want, including at least: survival months, vital status recode,
   and (optionally) a cause-of-death / SEER-cause-specific death classification
   and AJCC stage.
2. Export the case listing to CSV into ``data/raw/seer/``.
3. Point this loader at that CSV.

See ``data/DATA_SOURCES.md`` for the exact steps. As with all source data, SEER
exports are governed by their DUA and must never be committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vca._km import kaplan_meier

# Default column names as they commonly appear in SEER*Stat CSV exports. Override
# via ``column_map`` because exact labels depend on your SEER*Stat variable set.
DEFAULT_SEER_COLUMNS = {
    "survival_months": "Survival months",
    "vital_status": "Vital status recode (study cutoff used)",
    "cause_specific": "SEER cause-specific death classification",
    "stage": "Derived AJCC Stage Group, 7th ed (2010-2015)",
}
# Values that indicate the patient died (vital status recode).
DEAD_LABELS = {"Dead", "4", 4}
# Values that indicate death was attributable to this cancer (cause-specific).
CANCER_DEATH_LABELS = {"Dead (attributable to this cancer dx)", "1", 1}


@dataclass
class SeerSurvival:
    """A SEER-derived survival curve for external comparison."""

    time_days: np.ndarray       # KM step times (days)
    survival: np.ndarray        # S(t) at each step
    n: int                      # cohort size
    endpoint: str               # "overall" or "cause_specific"

    def survival_at(self, day: float) -> float:
        """S(day) by step-function lookup."""
        if self.time_days.size == 0:
            return 1.0
        idx = np.searchsorted(self.time_days, day, side="right") - 1
        return 1.0 if idx < 0 else float(self.survival[idx])


def load_seer_caselisting(
    path: str | Path, column_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Load a SEER*Stat case-listing CSV and normalise the columns we use."""
    cols = {**DEFAULT_SEER_COLUMNS, **(column_map or {})}
    df = pd.read_csv(path)
    present = {canon: src for canon, src in cols.items() if src in df.columns}
    if "survival_months" not in present or "vital_status" not in present:
        raise KeyError(
            "SEER export must contain survival-months and vital-status columns. "
            f"Looked for {cols['survival_months']!r} and {cols['vital_status']!r}; "
            f"got {list(df.columns)[:15]}..."
        )
    out = df.rename(columns={src: canon for canon, src in present.items()})
    keep = [c for c in ("survival_months", "vital_status", "cause_specific", "stage") if c in out.columns]
    out = out[keep].copy()
    out["survival_months"] = pd.to_numeric(out["survival_months"], errors="coerce")
    return out.dropna(subset=["survival_months"])


def seer_survival_curve(df: pd.DataFrame, endpoint: str = "overall") -> SeerSurvival:
    """Kaplan-Meier survival from a normalised SEER case listing.

    Parameters
    ----------
    endpoint
        ``"overall"`` counts any death as an event; ``"cause_specific"`` counts
        only cancer-attributable deaths (requires the cause-specific column).
    """
    time_days = df["survival_months"].to_numpy(float) * 30.4375
    if endpoint == "overall":
        event = df["vital_status"].isin(DEAD_LABELS).astype(int).to_numpy()
    elif endpoint == "cause_specific":
        if "cause_specific" not in df.columns:
            raise KeyError("cause_specific column required for endpoint='cause_specific'")
        event = df["cause_specific"].isin(CANCER_DEATH_LABELS).astype(int).to_numpy()
    else:
        raise ValueError("endpoint must be 'overall' or 'cause_specific'")

    t, s = kaplan_meier(time_days, event)
    return SeerSurvival(time_days=t, survival=s, n=len(df), endpoint=endpoint)


if __name__ == "__main__":  # pragma: no cover
    print("Usage: place a SEER*Stat CSV export in data/raw/seer/ and call")
    print("  df = load_seer_caselisting(path); curve = seer_survival_curve(df)")
