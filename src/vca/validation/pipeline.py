"""Held-out validation orchestration.

``run_validation`` fits a model (if needed), simulates virtual patients for the
held-out covariates, and scores the simulation against the real held-out
outcomes, producing a :class:`ValidationReport` that serialises to JSON/CSV and
(optionally) writes figures. ``compare_models`` runs several models through the
same harness so the Bayesian model can be judged against the baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vca.data_processing.schema import TrialData
from vca.models.base import SimulationResult, TrajectoryModel
from vca.validation import metrics as M
from vca.validation.survival import compare_survival

DEFAULT_EVENT_LANDMARKS = (180.0, 365.0)   # ~6 and ~12 months
DEFAULT_SLD_LANDMARK = 180.0
DEFAULT_COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


@dataclass
class ValidationReport:
    model_name: str
    n_train: int
    n_test: int
    n_draws: int
    endpoints: dict = field(default_factory=dict)
    sld: dict = field(default_factory=dict)
    headline: dict = field(default_factory=dict)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))
        return path

    def to_rows(self) -> pd.DataFrame:
        """Flatten the report into a tidy, citable metrics table."""
        rows = []
        for ep, d in self.endpoints.items():
            for t, lm in d.get("landmarks", {}).items():
                rows.append({
                    "model": self.model_name, "endpoint": ep, "metric": "calibration_ece",
                    "landmark_days": float(t), "value": lm["calibration_ece"],
                })
                rows.append({
                    "model": self.model_name, "endpoint": ep, "metric": "brier_ipcw",
                    "landmark_days": float(t), "value": lm["brier_ipcw"],
                })
            surv = d.get("survival", {})
            for k in ("logrank_p", "median_abs_diff_days"):
                rows.append({
                    "model": self.model_name, "endpoint": ep, "metric": k,
                    "landmark_days": None, "value": surv.get(k),
                })
        for t, s in self.sld.get("landmarks", {}).items():
            rows.append({
                "model": self.model_name, "endpoint": "sld", "metric": "crps",
                "landmark_days": float(t), "value": s["crps"],
            })
            rows.append({
                "model": self.model_name, "endpoint": "sld", "metric": "coverage_90",
                "landmark_days": float(t), "value": s["coverage"].get("0.9") or s["coverage"].get(0.9),
            })
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_rows().to_csv(path, index=False)
        return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def _aligned_events(test: TrialData, order) -> pd.DataFrame:
    """Test events reindexed to the simulation's patient order."""
    ev = test.events.set_index("patient_id")
    return ev.loc[[str(p) for p in order]]


def _observed_sld_at(test: TrialData, t: float, order) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each patient's observed SLD at landmark ``t``.

    Returns a boolean mask over ``order`` (patients whose measurements bracket
    ``t``) and the interpolated observed SLD for those patients.
    """
    lg = test.longitudinal
    mask = np.zeros(len(order), dtype=bool)
    values = np.full(len(order), np.nan)
    grouped = {str(pid): g for pid, g in lg.groupby("patient_id")}
    for i, pid in enumerate(order):
        g = grouped.get(str(pid))
        if g is None or len(g) == 0:
            continue
        gt = g["time_days"].to_numpy(float)
        gy = g["sld_mm"].to_numpy(float)
        if gt.min() <= t <= gt.max():
            values[i] = float(np.interp(t, gt, gy))
            mask[i] = True
    return mask, values


def run_validation(
    model: TrajectoryModel,
    train: TrialData,
    test: TrialData,
    *,
    n_draws: int = 300,
    event_landmarks=DEFAULT_EVENT_LANDMARKS,
    sld_landmarks=(DEFAULT_SLD_LANDMARK,),
    endpoints=("pfs", "os"),
    coverage_levels=DEFAULT_COVERAGE_LEVELS,
    seed: int = 0,
    out_dir: str | Path | None = None,
    make_figures: bool = False,
    model_name: str | None = None,
) -> ValidationReport:
    """Fit (if needed), simulate, and score a model on held-out data."""
    if not model.is_fitted:
        model.fit(train)
    name = model_name or type(model).__name__

    times = np.arange(0.0, 721.0, 30.0)
    result: SimulationResult = model.simulate(
        test.covariates(), n_draws=n_draws, times=times, seed=seed
    )
    order = list(result.covariates.index)
    ev = _aligned_events(test, order)

    report = ValidationReport(
        model_name=name, n_train=train.n_patients, n_test=test.n_patients, n_draws=n_draws
    )

    # -- event endpoints -----------------------------------------------------
    survival_cmps = {}
    for ep in endpoints:
        real_time = ev[f"{ep}_time_days"].to_numpy(float)
        real_event = ev[f"{ep}_event"].to_numpy(int)
        landmarks = {}
        for t in event_landmarks:
            pred = result.predicted_event_prob(ep, t)
            cal = M.landmark_calibration(pred, real_time, real_event, t)
            brier = M.brier_score_ipcw(pred, real_time, real_event, t)
            landmarks[float(t)] = {
                "calibration_ece": cal.calibration_error,
                "brier_ipcw": brier,
                "n_at_landmark": cal.n_used,
            }
        sim_latent, _ = result.sample_one_per_patient(ep, seed=seed + 7)
        cmp = compare_survival(ep, sim_latent, real_time, real_event, seed=seed + 11)
        survival_cmps[ep] = cmp
        report.endpoints[ep] = {"landmarks": landmarks, "survival": cmp.to_dict()}

    # -- SLD trajectory ------------------------------------------------------
    sld_landmark_metrics = {}
    for t in sld_landmarks:
        ti = int(np.argmin(np.abs(times - t)))
        pred_samples = result.sld[:, :, ti].T            # (n_pat, n_draws)
        mask, observed = _observed_sld_at(test, t, order)
        if mask.sum() >= 5:
            cov = M.coverage_table(pred_samples[mask], observed[mask], levels=coverage_levels)
            crps_val = M.crps(pred_samples[mask], observed[mask])
        else:
            cov, crps_val = {lev: float("nan") for lev in coverage_levels}, float("nan")
        sld_landmark_metrics[float(t)] = {
            "crps": crps_val,
            "coverage": {str(k): v for k, v in cov.items()},
            "n": int(mask.sum()),
        }
    report.sld = {"landmarks": sld_landmark_metrics}

    # -- headline summary ----------------------------------------------------
    report.headline = {
        "mean_logrank_p": float(np.nanmean([c.logrank_p for c in survival_cmps.values()])),
        "mean_brier_ipcw": float(np.nanmean([
            lm["brier_ipcw"] for ep in report.endpoints.values()
            for lm in ep["landmarks"].values()
        ])),
        "mean_calibration_ece": float(np.nanmean([
            lm["calibration_ece"] for ep in report.endpoints.values()
            for lm in ep["landmarks"].values()
        ])),
        "sld_crps_primary": sld_landmark_metrics[float(sld_landmarks[0])]["crps"],
    }

    # -- outputs -------------------------------------------------------------
    if out_dir is not None:
        out = Path(out_dir)
        report.to_json(out / f"{name}_validation.json")
        report.to_csv(out / f"{name}_validation.csv")
        if make_figures:
            _write_figures(report, result, survival_cmps, test, times, sld_landmarks,
                           order, ev, out / "figures", name, seed)
    return report


def _write_figures(report, result, survival_cmps, test, times, sld_landmarks,
                   order, ev, fig_dir, name, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vca.viz import plots

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # KM overlays + calibration per endpoint.
    for ep, cmp in survival_cmps.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plots.plot_km_overlay(cmp, ax=axes[0])
        t0 = float(list(report.endpoints[ep]["landmarks"])[0])
        pred = result.predicted_event_prob(ep, t0)
        cal = M.landmark_calibration(
            pred, ev[f"{ep}_time_days"].to_numpy(float),
            ev[f"{ep}_event"].to_numpy(int), t0,
        )
        plots.plot_calibration(cal, ax=axes[1], title=f"{ep.upper()} calibration @ {t0:.0f}d")
        fig.suptitle(f"{name} — {ep.upper()}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}_{ep}_survival.png", dpi=120)
        plt.close(fig)

    # SLD trajectory example (first patient with enough observed points).
    grouped = {str(pid): g for pid, g in test.longitudinal.groupby("patient_id")}
    for j, pid in enumerate(order):
        g = grouped.get(str(pid))
        if g is not None and len(g) >= 3:
            fig, ax = plt.subplots(figsize=(5, 4))
            plots.plot_trajectory_bands(
                times, result.sld[:, j, :],
                observed_times=g["time_days"].to_numpy(float),
                observed_sld=g["sld_mm"].to_numpy(float),
                ax=ax, title=f"{name} — SLD, patient {pid}",
            )
            fig.tight_layout()
            fig.savefig(fig_dir / f"{name}_sld_example.png", dpi=120)
            plt.close(fig)
            break


def compare_models(
    models: dict[str, TrajectoryModel],
    train: TrialData,
    test: TrialData,
    *,
    out_dir: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run several models through ``run_validation`` and return a merged table."""
    frames, headlines = [], []
    for name, model in models.items():
        rep = run_validation(model, train, test, out_dir=out_dir, model_name=name, **kwargs)
        frames.append(rep.to_rows())
        headlines.append({"model": name, **rep.headline})
    table = pd.concat(frames, ignore_index=True)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "model_comparison.csv", index=False)
        pd.DataFrame(headlines).to_csv(out / "headline_comparison.csv", index=False)
    return pd.DataFrame(headlines)
