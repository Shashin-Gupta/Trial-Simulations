"""Canonical data schema for the virtual control arm.

Every data source (Project Data Sphere exports, synthetic data, SEER-derived
calibration sets) is mapped onto the *same* three-table schema so that models
and validation code never depend on a particular source's column names. This is
the contract that lets a more complex model be swapped in later without
rewriting the pipeline.

The three tables
----------------
1. **baseline**    — one row per patient; time-invariant covariates.
2. **longitudinal**— long format; repeated RECIST tumour-size measurements.
3. **events**      — one row per patient; time-to-event outcomes (PFS, OS).

They are joined on ``patient_id``. A :class:`TrialData` bundles them together
and validates their mutual consistency.

Units & conventions
--------------------
- Time is measured in **days from randomisation/baseline**; baseline is ``t = 0``.
- Tumour size is the RECIST 1.1 **sum of longest diameters (SLD) in millimetres**
  of target lesions.
- Event indicators are ``1`` = event observed, ``0`` = right-censored.
- ``pfs_event`` counts progression *or* death; ``os_event`` counts death.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column contracts
# --------------------------------------------------------------------------- #

PATIENT_ID = "patient_id"

# baseline table -------------------------------------------------------------
BASELINE_REQUIRED: list[str] = [
    PATIENT_ID,
    "age",              # years, float
    "sex",              # {"M", "F"}
    "ecog_ps",          # ECOG performance status, integer 0-4
    "stage",            # AJCC stage string, e.g. "IV", "IIIB"
    "histology",        # {"squamous", "non_squamous", "other"}
    "smoking",          # {"current", "former", "never", "unknown"}
    "prior_lines",      # number of prior systemic therapy lines, integer >= 0
    "baseline_sld_mm",  # SLD of target lesions at baseline, mm
    "n_target_lesions", # integer >= 0
]
# Optional covariates — frequently missing, especially in pre-biomarker-era
# trials. Loaders should include them when available; models must tolerate NaN.
BASELINE_OPTIONAL: list[str] = [
    "study_id",         # source trial identifier
    "treatment",        # comparator regimen label
    "egfr_status",      # {"mutant", "wildtype", "unknown"}
    "alk_status",       # {"positive", "negative", "unknown"}
    "pdl1_tps",         # PD-L1 tumour proportion score, 0-100 (%)
]

# longitudinal table ---------------------------------------------------------
LONGITUDINAL_REQUIRED: list[str] = [PATIENT_ID, "time_days", "sld_mm"]

# events table ---------------------------------------------------------------
EVENTS_REQUIRED: list[str] = [
    PATIENT_ID,
    "pfs_time_days",
    "pfs_event",
    "os_time_days",
    "os_event",
]

# Controlled vocabularies used by loaders and the synthetic generator.
SEX_LEVELS = ("M", "F")
HISTOLOGY_LEVELS = ("squamous", "non_squamous", "other")
SMOKING_LEVELS = ("current", "former", "never", "unknown")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class SchemaError(ValueError):
    """Raised when a table violates the canonical schema."""


def _require_columns(df: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{table} table is missing required columns: {missing}")


def validate_trial_data(td: TrialData, *, strict: bool = True) -> TrialData:
    """Validate a :class:`TrialData` against the canonical schema.

    Checks column presence, event-indicator domains, non-negative times and
    tumour sizes, unique baseline ``patient_id``, and referential integrity
    across the three tables.

    Parameters
    ----------
    strict
        If ``True`` (default), raise on any violation. If ``False``, only raise
        on structural problems (missing columns / duplicate ids) and let value
        issues pass — useful when profiling messy raw data.
    """
    _require_columns(td.baseline, BASELINE_REQUIRED, "baseline")
    _require_columns(td.longitudinal, LONGITUDINAL_REQUIRED, "longitudinal")
    _require_columns(td.events, EVENTS_REQUIRED, "events")

    if td.baseline[PATIENT_ID].duplicated().any():
        dups = td.baseline.loc[td.baseline[PATIENT_ID].duplicated(), PATIENT_ID].tolist()
        raise SchemaError(f"Duplicate patient_id in baseline table: {dups[:5]}")

    if td.events[PATIENT_ID].duplicated().any():
        raise SchemaError("Duplicate patient_id in events table (expected one row per patient)")

    base_ids = set(td.baseline[PATIENT_ID])
    event_ids = set(td.events[PATIENT_ID])
    long_ids = set(td.longitudinal[PATIENT_ID])

    if not event_ids.issubset(base_ids):
        raise SchemaError("events table references patient_id values absent from baseline")
    if not long_ids.issubset(base_ids):
        raise SchemaError("longitudinal table references patient_id values absent from baseline")

    if strict:
        for col in ("pfs_event", "os_event"):
            bad = set(pd.unique(td.events[col].dropna())) - {0, 1}
            if bad:
                raise SchemaError(f"events.{col} must be in {{0, 1}}; found {sorted(bad)}")
        for col in ("pfs_time_days", "os_time_days"):
            if (td.events[col] < 0).any():
                raise SchemaError(f"events.{col} contains negative times")
        if (td.longitudinal["time_days"] < 0).any():
            raise SchemaError("longitudinal.time_days contains negative times")
        if (td.longitudinal["sld_mm"] < 0).any():
            raise SchemaError("longitudinal.sld_mm contains negative tumour sizes")

    return td


def coerce_dtypes(td: TrialData) -> TrialData:
    """Best-effort coercion of the three tables to canonical dtypes (in place)."""
    b, e, lo = td.baseline, td.events, td.longitudinal
    for col in ("age", "baseline_sld_mm", "pdl1_tps"):
        if col in b:
            b[col] = pd.to_numeric(b[col], errors="coerce")
    for col in ("ecog_ps", "prior_lines", "n_target_lesions"):
        if col in b:
            b[col] = pd.to_numeric(b[col], errors="coerce").astype("Int64")
    for col in ("pfs_time_days", "os_time_days"):
        e[col] = pd.to_numeric(e[col], errors="coerce")
    for col in ("pfs_event", "os_event"):
        e[col] = pd.to_numeric(e[col], errors="coerce").astype("Int64")
    lo["time_days"] = pd.to_numeric(lo["time_days"], errors="coerce")
    lo["sld_mm"] = pd.to_numeric(lo["sld_mm"], errors="coerce")
    for col in (PATIENT_ID,):
        b[col] = b[col].astype(str)
        e[col] = e[col].astype(str)
        lo[col] = lo[col].astype(str)
    return td


# --------------------------------------------------------------------------- #
# TrialData container
# --------------------------------------------------------------------------- #

@dataclass
class TrialData:
    """A validated bundle of the three canonical tables.

    Attributes
    ----------
    baseline
        One row per patient, indexed positionally; must contain
        :data:`BASELINE_REQUIRED`.
    longitudinal
        Long-format repeated tumour measurements; :data:`LONGITUDINAL_REQUIRED`.
    events
        One row per patient of time-to-event outcomes; :data:`EVENTS_REQUIRED`.
    """

    baseline: pd.DataFrame
    longitudinal: pd.DataFrame
    events: pd.DataFrame

    # -- construction --------------------------------------------------------
    def __post_init__(self) -> None:
        # Defensive copies so downstream mutation never surprises the caller.
        self.baseline = self.baseline.reset_index(drop=True).copy()
        self.longitudinal = self.longitudinal.reset_index(drop=True).copy()
        self.events = self.events.reset_index(drop=True).copy()

    def validate(self, *, strict: bool = True) -> TrialData:
        return validate_trial_data(self, strict=strict)

    # -- introspection -------------------------------------------------------
    @property
    def patient_ids(self) -> np.ndarray:
        return self.baseline[PATIENT_ID].to_numpy()

    @property
    def n_patients(self) -> int:
        return len(self.baseline)

    @property
    def covariate_columns(self) -> list[str]:
        """Baseline columns usable as model covariates (excludes identifiers)."""
        drop = {PATIENT_ID, "study_id", "treatment"}
        return [c for c in self.baseline.columns if c not in drop]

    def covariates(self) -> pd.DataFrame:
        """Return the baseline covariate matrix, indexed by ``patient_id``."""
        return self.baseline.set_index(PATIENT_ID)[
            [c for c in self.covariate_columns if c in self.baseline.columns]
        ]

    # -- subsetting / splitting ---------------------------------------------
    def subset(self, patient_ids: Iterable[str]) -> TrialData:
        """Return a new :class:`TrialData` restricted to ``patient_ids``."""
        ids = set(map(str, patient_ids))
        return TrialData(
            baseline=self.baseline[self.baseline[PATIENT_ID].astype(str).isin(ids)],
            longitudinal=self.longitudinal[self.longitudinal[PATIENT_ID].astype(str).isin(ids)],
            events=self.events[self.events[PATIENT_ID].astype(str).isin(ids)],
        )

    def train_test_split(
        self, test_fraction: float = 0.3, *, seed: int = 0
    ) -> tuple[TrialData, TrialData]:
        """Split patients (not rows) into train/test partitions.

        Splitting on patients — never on individual measurements — is essential:
        leaking a patient's early tumour measurements into training while holding
        out their later ones would massively inflate apparent performance.
        """
        rng = np.random.default_rng(seed)
        ids = self.patient_ids.copy()
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_fraction))
        test_ids, train_ids = ids[:n_test], ids[n_test:]
        return self.subset(train_ids), self.subset(test_ids)

    # -- io ------------------------------------------------------------------
    def to_parquet(self, directory, prefix: str = "nsclc") -> None:
        """Write the three tables to ``directory`` as Parquet (kept local)."""
        from pathlib import Path

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.baseline.to_parquet(d / f"{prefix}_baseline.parquet")
        self.longitudinal.to_parquet(d / f"{prefix}_longitudinal.parquet")
        self.events.to_parquet(d / f"{prefix}_events.parquet")

    @classmethod
    def from_parquet(cls, directory, prefix: str = "nsclc") -> TrialData:
        from pathlib import Path

        d = Path(directory)
        return cls(
            baseline=pd.read_parquet(d / f"{prefix}_baseline.parquet"),
            longitudinal=pd.read_parquet(d / f"{prefix}_longitudinal.parquet"),
            events=pd.read_parquet(d / f"{prefix}_events.parquet"),
        ).validate(strict=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TrialData(n_patients={self.n_patients}, "
            f"n_measurements={len(self.longitudinal)}, "
            f"pfs_events={int(self.events['pfs_event'].sum())}, "
            f"os_events={int(self.events['os_event'].sum())})"
        )
