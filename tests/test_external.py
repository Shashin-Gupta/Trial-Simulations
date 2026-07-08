"""Unit tests for the external aggregate-validation building blocks.

These are pure-logic tests (no MCMC, no real data): RECIST BOR classification
from crafted trajectories, and the BOR contingency test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vca.data_processing.pds_trials import BOR_LEVELS
from vca.validation.external import _bor_test, classify_bor

TIMES = np.arange(0.0, 541.0, 30.0)


def test_pure_responder_is_pr():
    # shrink to ~60% of baseline: >=30% decrease (PR) but not near-zero (CR)
    sld = np.linspace(100.0, 62.0, len(TIMES))
    assert classify_bor(TIMES, sld) == "PR"


def test_near_complete_response_is_cr():
    sld = np.full_like(TIMES, 3.0)  # ~0 mm throughout
    sld[0] = 80.0
    assert classify_bor(TIMES, sld) == "CR"


def test_stable_disease():
    sld = 100.0 + 5.0 * np.sin(TIMES / 100.0)  # wobbles < 20%, never -30%
    assert classify_bor(TIMES, sld) == "SD"


def test_early_progressor_is_pd():
    base = 100.0
    sld = base * np.exp(0.01 * TIMES)  # grows from t0 -> >20% by first visit
    assert classify_bor(TIMES, sld) == "PD"


def test_stable_then_progress_is_sd_not_pd():
    # flat for a while (SD), then late regrowth beyond +20% of nadir
    sld = np.concatenate([np.full(6, 100.0), np.linspace(100.0, 200.0, len(TIMES) - 6)])
    assert classify_bor(TIMES, sld) == "SD"


def test_respond_then_progress_is_pr():
    # deep response then regrowth -> best response wins (PR)
    down = np.linspace(100.0, 50.0, 6)          # -50% -> PR territory
    up = np.linspace(50.0, 150.0, len(TIMES) - 6)
    assert classify_bor(TIMES, np.concatenate([down, up])) == "PR"


def test_bor_test_detects_difference_and_agreement():
    same = pd.Series([50, 50, 50, 50], index=list(BOR_LEVELS))
    diff = pd.Series([5, 5, 5, 135], index=list(BOR_LEVELS))
    p_same = _bor_test(same, same.copy())["p_value"]
    p_diff = _bor_test(same, diff)["p_value"]
    assert p_same > 0.9        # identical tables -> not different
    assert p_diff < 0.001      # very different -> detected
