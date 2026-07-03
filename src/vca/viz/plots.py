"""Matplotlib helpers for validation and exploration figures.

Functions accept an optional ``ax`` and return it, so they compose into
multi-panel figures. The library does not force a backend; headless callers
(e.g. ``scripts/run_demo.py``) should set ``matplotlib.use("Agg")`` first.
"""

from __future__ import annotations

import numpy as np

from vca.validation.metrics import CalibrationCurve
from vca.validation.survival import SurvivalComparison


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(5, 4))
    return ax


def plot_calibration(cal: CalibrationCurve, ax=None, title: str = "Calibration"):
    """Reliability diagram: observed vs predicted event probability."""
    ax = _ax(ax)
    valid = cal.bin_count > 0
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="ideal")
    ax.scatter(
        cal.bin_pred[valid], cal.bin_obs[valid],
        s=20 + 3 * cal.bin_count[valid], alpha=0.8, label="model",
    )
    ax.set_xlabel("Predicted P(event by t)")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title} (ECE={cal.calibration_error:.3f}, n={cal.n_used})")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_km_overlay(cmp: SurvivalComparison, ax=None):
    """Overlay real vs simulated Kaplan-Meier curves with CI bands."""
    ax = _ax(ax)
    for km, color, name in (
        (cmp.real_km, "C0", "real"),
        (cmp.sim_km, "C1", "simulated"),
    ):
        ax.step(km.timeline, km.survival, where="post", color=color, label=name)
        ax.fill_between(km.timeline, km.ci_lower, km.ci_upper, step="post", color=color, alpha=0.15)
    ax.set_xlabel("Days")
    ax.set_ylabel(f"S(t) — {cmp.endpoint.upper()}")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{cmp.endpoint.upper()}: log-rank p={cmp.logrank_p:.2f}, "
                 f"median Δ={cmp.median_abs_diff_days:.0f}d")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_coverage(coverage: dict[float, float], ax=None, title="Interval coverage"):
    """Nominal vs empirical prediction-interval coverage."""
    ax = _ax(ax)
    levels = sorted(coverage)
    emp = [coverage[l] for l in levels]
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="ideal")
    ax.plot(levels, emp, "o-", label="empirical")
    ax.set_xlabel("Nominal level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_trajectory_bands(times, sld_samples, observed_times=None, observed_sld=None,
                          ax=None, title="Simulated SLD trajectory"):
    """Predictive SLD median + 90% band for one patient, with observed points.

    ``sld_samples`` is (n_draws, n_times) for a single patient.
    """
    ax = _ax(ax)
    med = np.nanmedian(sld_samples, axis=0)
    lo = np.nanquantile(sld_samples, 0.05, axis=0)
    hi = np.nanquantile(sld_samples, 0.95, axis=0)
    ax.fill_between(times, lo, hi, alpha=0.2, color="C1", label="90% PI")
    ax.plot(times, med, color="C1", label="predictive median")
    if observed_times is not None and observed_sld is not None:
        ax.scatter(observed_times, observed_sld, color="C0", s=25, zorder=5, label="observed")
    ax.set_xlabel("Days")
    ax.set_ylabel("SLD (mm)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax
