#!/usr/bin/env python
"""Steps 2 + 3: train on real _438, internally validate, externally validate.

Fits the hierarchical Bayesian TGI+survival model (and the resampling baseline)
on the real _438 training split, scores them on the held-out _438 20%, then runs
the external aggregate validation against the four BOR/survival-only trials with
the _438-trained model. Writes machine-readable metrics + figures under
``results/real_data/`` (git-ignored; may reference real patients).

    python scripts/run_real_data_validation.py                    # full run
    python scripts/run_real_data_validation.py --quick            # tiny MCMC smoke
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vca.data_processing.pds_trials import VALIDATION_TRIALS, load_438, load_trial
from vca.data_processing.schema import TrialData
from vca.models.baseline import MarginalResamplingModel
from vca.validation.external import external_validate_trial
from vca.validation.pipeline import run_validation
from vca.viz import plots

warnings.filterwarnings("ignore")
DAYS_PER_MONTH = 30.4375
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real_data"


def measurable_cohort(td: TrialData) -> TrialData:
    """Restrict _438 to patients with a baseline target-lesion SLD (TGI-modelable)."""
    ok = pd.to_numeric(td.baseline["baseline_sld_mm"], errors="coerce").notna()
    return td.subset(td.baseline.loc[ok, "patient_id"].tolist())


def stratified_split(td: TrialData, *, test_fraction=0.2, seed=0, by="histology"):
    """Patient-level split stratified by ``by`` (falls back to random if too sparse)."""
    rng = np.random.default_rng(seed)
    base = td.baseline
    strata = base[by].astype("string").fillna("NA")
    test_ids: list[str] = []
    ok = True
    for _level, grp in base.groupby(strata):
        ids = grp["patient_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        k = int(round(len(ids) * test_fraction))
        if len(ids) < 5:
            ok = False
        test_ids.extend(ids[:k].tolist())
    if not ok:  # a stratum too small to split sensibly -> plain random split
        return td.train_test_split(test_fraction, seed=seed), "random"
    train_ids = [p for p in base["patient_id"].astype(str) if p not in set(test_ids)]
    return (td.subset(train_ids), td.subset(test_ids)), f"stratified_by_{by}"


def _median_mo(rep_surv: dict, key: str) -> float | None:
    v = rep_surv.get(key)
    return None if v is None else round(v / DAYS_PER_MONTH, 2)


# --------------------------------------------------------------------------- #
# Step 2 — internal validation on _438 held-out
# --------------------------------------------------------------------------- #

def run_internal(args) -> dict:
    td = measurable_cohort(load_438().data)
    (train, test), split_kind = stratified_split(
        td, test_fraction=args.test_fraction, seed=args.seed)
    print(f"[Step 2] _438 modelable n={td.n_patients}  "
          f"train={train.n_patients} test={test.n_patients}  split={split_kind}")

    from vca.models.tgi_survival import TGISurvivalModel
    bayes = TGISurvivalModel(num_warmup=args.num_warmup, num_samples=args.num_samples,
                             num_chains=args.num_chains, seed=args.seed)
    print("[Step 2] fitting Bayesian TGI+survival model on real _438 train split...")
    bayes.fit(train)

    internal_dir = OUT
    internal_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    models = {"bayesian_tgi_survival": bayes,
              "baseline_resampling": MarginalResamplingModel(min_donors=6)}
    for name, model in models.items():
        rep = run_validation(model, train, test, n_draws=args.n_draws, seed=args.seed,
                             out_dir=internal_dir, make_figures=(name == "bayesian_tgi_survival"),
                             model_name=f"438_internal_{name}")
        reports[name] = rep
        print(f"  {name}: logrank_p(mean)={rep.headline['mean_logrank_p']:.3f} "
              f"brier={rep.headline['mean_brier_ipcw']:.3f} "
              f"ece={rep.headline['mean_calibration_ece']:.3f} "
              f"sld_crps={rep.headline['sld_crps_primary']:.2f}")

    _write_internal_vs_synthetic(reports["bayesian_tgi_survival"], split_kind)
    return {"bayes_model": bayes, "train": train, "test": test,
            "report": reports["bayesian_tgi_survival"]}


def _write_internal_vs_synthetic(real_rep, split_kind) -> None:
    """Compare real-data internal metrics to the prior synthetic-data metrics."""
    syn_path = ROOT / "results" / "bayesian_tgi_survival_validation.json"
    rows = []
    real_h = real_rep.headline
    syn_h = json.loads(syn_path.read_text())["headline"] if syn_path.exists() else {}
    for k, better in [("mean_logrank_p", "higher"), ("mean_brier_ipcw", "lower"),
                      ("mean_calibration_ece", "lower"), ("sld_crps_primary", "lower")]:
        rows.append({"metric": k, "better": better,
                     "synthetic": round(syn_h.get(k, float("nan")), 4) if syn_h else None,
                     "real_438": round(real_h[k], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "internal_vs_synthetic.csv", index=False)
    (OUT / "internal_vs_synthetic.md").write_text(
        f"# Internal validation: real _438 vs synthetic\n\n"
        f"_438 split: {split_kind}. Higher log-rank p = curves indistinguishable "
        f"(good); lower Brier/ECE/CRPS = better.\n\n" + df.to_markdown(index=False) + "\n")
    print("\n[Step 2] real vs synthetic headline:")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------- #
# Step 3 — external aggregate validation
# --------------------------------------------------------------------------- #

def run_external(bayes_model, donor: TrialData, args) -> None:
    ext_dir = OUT / "external"
    fig_dir = ext_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n[Step 3] external aggregate validation:")
    for tid in VALIDATION_TRIALS:
        real = load_trial(tid)
        res, sim, surv_cmps = external_validate_trial(
            bayes_model, real, donor, n_draws=args.n_draws, seed=args.seed)
        (ext_dir / f"{tid}_external.json").write_text(json.dumps(res.to_dict(), indent=2))
        _plot_external(tid, real, res, surv_cmps, fig_dir)

        row = {"trial_id": tid, "regimen": res.regimen, "line": res.line,
               "histology": res.histology_label, "n_real": res.n_real,
               "bor_test": res.bor.get("test"), "bor_p": res.bor.get("p_value"),
               "pfs_logrank_p": (res.pfs or {}).get("logrank_p"),
               "pfs_real_mo": (res.pfs or {}).get("real_median_months"),
               "pfs_sim_mo": (res.pfs or {}).get("sim_median_months"),
               "os_logrank_p": (res.os or {}).get("logrank_p"),
               "os_real_mo": (res.os or {}).get("real_median_months"),
               "os_sim_mo": (res.os or {}).get("sim_median_months")}
        rows.append(row)
        print(f"  {tid} ({res.regimen}, {res.line}, {res.histology_label}): "
              f"BOR p={_fmt(row['bor_p'])}  PFS logrank p={_fmt(row['pfs_logrank_p'])}  "
              f"OS logrank p={_fmt(row['os_logrank_p'])}")

    table = pd.DataFrame(rows)
    table.to_csv(ext_dir / "external_summary.csv", index=False)
    (ext_dir / "external_summary.md").write_text(
        "# External aggregate validation (per trial)\n\n"
        "Large p-values = simulated and real aggregates are not detectably "
        "different (desirable). PFS omitted where not derivable (133).\n\n"
        + table.to_markdown(index=False) + "\n")
    print(f"\nWrote external results to {ext_dir}")


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def _plot_external(tid, real, res, surv_cmps, fig_dir) -> None:
    # KM overlays (sim vs real) for available endpoints
    for ep, cmp in surv_cmps.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        plots.plot_km_overlay(cmp, ax=ax)
        ax.set_title(f"Trial {tid} — {ep.upper()} (sim vs real)\n{res.regimen}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{tid}_{ep}_km.png", dpi=120)
        plt.close(fig)
    # BOR grouped bars
    rp = [res.bor["real_proportions"].get(c, 0) for c in ["CR", "PR", "SD", "PD"]]
    sp = [res.bor["sim_proportions"].get(c, 0) for c in ["CR", "PR", "SD", "PD"]]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - 0.2, rp, 0.4, label="real")
    ax.bar(x + 0.2, sp, 0.4, label="simulated")
    ax.set_xticks(x)
    ax.set_xticklabels(["CR", "PR", "SD", "PD"])
    ax.set_ylabel("proportion (evaluable)")
    ax.set_title(f"Trial {tid} — BOR (p={_fmt(res.bor.get('p_value'))})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{tid}_bor.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--n-draws", type=int, default=300)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--num-chains", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny MCMC for a smoke run")
    args = p.parse_args(argv)
    if args.quick:
        args.num_warmup = args.num_samples = 80
        args.num_chains = 1
        args.n_draws = 150

    internal = run_internal(args)
    donor = measurable_cohort(load_438().data)
    run_external(internal["bayes_model"], donor, args)
    print(f"\nAll real-data validation outputs under {OUT.resolve()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
