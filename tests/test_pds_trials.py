"""Smoke tests for the real Project Data Sphere loaders.

The raw patient-level SAS files are DUA-protected and never committed, so these
tests skip cleanly on any machine that does not have the data placed under
``data/raw/<trial_id>/``. Where the data *is* present they assert the loaders
produce schema-valid canonical tables with plausible outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vca.data_processing import pds_trials
from vca.data_processing.pds_trials import BOR_LEVELS, TRIAL_DIRS

ALL_TRIALS = list(TRIAL_DIRS)


def _have(trial_id: str) -> bool:
    d = pds_trials._dir(trial_id)
    return d.exists() and any(d.glob("*.sas7bdat"))


pytestmark = pytest.mark.skipif(
    not all(_have(t) for t in ALL_TRIALS),
    reason="real PDS data not present under data/raw/ (DUA-protected)",
)


@pytest.mark.parametrize("trial_id", ALL_TRIALS)
def test_loader_produces_valid_schema(trial_id):
    rt = pds_trials.load_trial(trial_id)
    rt.data.validate(strict=False)                      # raises on structural issues
    assert rt.n > 50
    # baseline is one row per patient, ids unique
    assert rt.data.baseline["patient_id"].is_unique
    # OS is present and has some events for every trial
    os_e = pd.to_numeric(rt.data.events["os_event"], errors="coerce")
    assert os_e.sum() > 0
    # BOR is one label per patient in the controlled vocabulary
    assert set(rt.bor.unique()).issubset(set(BOR_LEVELS) | {"NE"})
    assert len(rt.bor) == rt.n


def test_438_has_longitudinal_others_do_not():
    t438 = pds_trials.load_438()
    assert len(t438.data.longitudinal) > 0
    assert t438.data.longitudinal["time_days"].min() >= -60
    base_sld = pd.to_numeric(t438.data.baseline["baseline_sld_mm"], errors="coerce")
    assert base_sld.dropna().between(1, 500).mean() > 0.9  # plausible mm range
    for tid in pds_trials.VALIDATION_TRIALS:
        assert len(pds_trials.load_trial(tid).data.longitudinal) == 0


def test_272_is_squamous_and_133_pfs_absent():
    t272 = pds_trials.load_272()
    assert (t272.data.baseline["histology"] == "squamous").all()
    t133 = pds_trials.load_133()
    assert t133.meta["pfs_available"] is False
    assert pd.to_numeric(t133.data.events["pfs_event"], errors="coerce").notna().sum() == 0


def test_os_medians_in_clinically_plausible_range():
    # every control arm should have a median OS roughly 6-18 months
    from vca._km import kaplan_meier
    for tid in ALL_TRIALS:
        ev = pds_trials.load_trial(tid).data.events
        t = pd.to_numeric(ev["os_time_days"], errors="coerce").to_numpy(float)
        e = pd.to_numeric(ev["os_event"], errors="coerce").to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(e)
        tt, ss = kaplan_meier(t[ok], e[ok].astype(int))
        below = tt[ss <= 0.5]
        med_months = (below[0] if below.size else np.nan) / 30.4375
        assert 5 < med_months < 20, f"{tid}: implausible median OS {med_months:.1f} mo"
