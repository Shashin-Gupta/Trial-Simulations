#!/usr/bin/env python
"""Export the manuscript figures as submission-ready files (300 DPI PNG + vector
PDF), composed into proper multi-panel figures with distributional annotations.

The KM/BOR panels show the canonical *representative replicate* curve (rebuilt by
re-fitting the deterministic canonical model, n_draws=400 — verified bit-exact
against results/real_data/*_validation.json), annotated with the DEPLOYED-estimand
posterior-predictive DISTRIBUTION from results/real_data/robustness_summary.json
(median p, 5-95% interval, significance rate) rather than a single p-value. Fig 5
visualises those p-value distributions directly. Fig 4 reuses
scripts/plot_fig4_landmark_reversal.py.

Palette: one validated colourblind-safe Okabe-Ito pair (blue=real, orange=sim).
Requires the DUA-protected data under data/raw/; outputs to
results/submission_figures/ (git-ignored).

    .venv/bin/python scripts/export_submission_figures.py
    .venv/bin/python scripts/export_submission_figures.py --quick   # smoke only
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_fig4_landmark_reversal import draw as draw_landmark  # noqa: E402
from run_real_data_validation import measurable_cohort, stratified_split  # noqa: E402
from vca.data_processing.pds_trials import VALIDATION_TRIALS, load_438, load_trial  # noqa: E402
from vca.models.tgi_survival import TGISurvivalModel  # noqa: E402
from vca.validation.external import external_validate_trial  # noqa: E402
from vca.validation.pipeline import _aligned_events  # noqa: E402
from vca.validation.survival import compare_survival  # noqa: E402
from vca.viz import plots  # noqa: E402

warnings.filterwarnings("ignore")

OUTDIR = ROOT / "results" / "submission_figures"
RESULTS = ROOT / "results" / "real_data"
REAL, SIM = "#0072B2", "#E69F00"          # Okabe-Ito, validated CVD-safe
THRESH = "#D55E00"                        # 0.05 threshold line
TRIAL_ORDER = ["141", "272", "133", "108"]
REGIMEN_SHORT = {"141": "Pac+Carbo+Bev", "272": "Gem+Cis",
                 "133": "Placebo+Docetaxel", "108": "Pac+Carbo"}
ROB = json.loads((RESULTS / "robustness_summary.json").read_text()) if \
    (RESULTS / "robustness_summary.json").exists() else None


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.prop_cycle": cycler(color=[REAL, SIM, "#009E73", "#CC79A7"]),
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, stem: str) -> list[Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def _rob(source, trial, ep):
    if ROB is None:
        return None
    d = ROB["internal"] if source == "internal" else ROB["external"].get(trial, {})
    return d.get(ep, {}).get("deployed")


def annot(source, trial, ep, *, fallback_p=None):
    """Distributional annotation string (DEPLOYED estimand)."""
    e = _rob(source, trial, ep)
    if e is None:
        return "" if fallback_p is None else f"p={fallback_p:.2f}"
    sig = e["sig_rate"]
    lab = f"{100 * (1 - sig):.0f}% n.s." if sig < 0.5 else f"{100 * sig:.0f}% sig"
    if ep == "bor":
        return f"median p={e['median_p']:.0e}; {lab}"
    return f"median p={e['median_p']:.2f} (5–95% {e['p5']:.2f}–{e['p95']:.2f}); {lab}"


# --------------------------------------------------------------------------- #
# rebuild the representative-replicate comparisons (verified vs committed JSON)
# --------------------------------------------------------------------------- #

def fit_pipeline(args):
    td = measurable_cohort(load_438().data)
    (train, test), split_kind = stratified_split(td, test_fraction=0.2, seed=args.seed)
    print(f"[fit] canonical {args.num_warmup}/{args.num_samples}/{args.num_chains} "
          f"seed={args.seed}; test={test.n_patients} split={split_kind}")
    model = TGISurvivalModel(num_warmup=args.num_warmup, num_samples=args.num_samples,
                             num_chains=args.num_chains, seed=args.seed)
    model.fit(train)
    return model, train, test


def internal_comparisons(model, test, args):
    times = np.arange(0.0, 721.0, 30.0)
    result = model.simulate(test.covariates(), n_draws=args.n_draws, times=times, seed=args.seed)
    order = list(result.covariates.index)
    ev = _aligned_events(test, order)
    cmps = {}
    for ep in ("pfs", "os"):
        real_time = ev[f"{ep}_time_days"].to_numpy(float)
        real_event = ev[f"{ep}_event"].to_numpy(int)
        sim_latent, _ = result.sample_one_per_patient(ep, seed=args.seed + 7)
        cmps[ep] = compare_survival(ep, sim_latent, real_time, real_event, seed=args.seed + 11)
    grouped = {str(pid): g for pid, g in test.longitudinal.groupby("patient_id")}
    sld = None
    for j, pid in enumerate(order):
        g = grouped.get(str(pid))
        if g is not None and len(g) >= 3:
            sld = dict(times=times, samples=result.sld[:, j, :], pid=pid,
                       obs_t=g["time_days"].to_numpy(float), obs_sld=g["sld_mm"].to_numpy(float))
            break
    return cmps, sld, test.n_patients


def external_all(model, donor, args):
    out = {}
    for tid in VALIDATION_TRIALS:
        real = load_trial(tid)
        res, _sim, surv = external_validate_trial(model, real, donor, n_draws=args.n_draws, seed=args.seed)
        out[str(tid)] = dict(res=res, surv=surv)
        print(f"[ext] {tid}: BOR p={res.bor.get('p_value'):.2g}  OS p={(res.os or {}).get('logrank_p'):.3f}")
    return out


def verify(internal_cmps, ext, tol=1e-6) -> bool:
    """Representative curves must reproduce the committed JSON bit-exactly, and the
    annotation sig-rates must match robustness_summary.json."""
    ok = True
    ij = json.loads((RESULTS / "438_internal_bayesian_tgi_survival_validation.json").read_text())
    print("\n[verify] representative replicate vs committed JSON (bit-exact):")
    for ep in ("pfs", "os"):
        got = internal_cmps[ep].logrank_p
        want = ij["endpoints"][ep]["survival"]["logrank_p"]
        bad = abs(got - want) > tol
        ok &= not bad
        print(f"  internal {ep.upper():3s}: {got:.6g} vs {want:.6g}{'  <-- DRIFT' if bad else ''}")
    for tid in TRIAL_ORDER:
        ej = json.loads((RESULTS / "external" / f"{tid}_external.json").read_text())
        for ep in ("pfs", "os"):
            if ej.get(ep) is None or ep not in ext[tid]["surv"]:
                continue
            got = ext[tid]["surv"][ep].logrank_p
            want = ej[ep]["logrank_p"]
            bad = abs(got - want) > tol
            ok &= not bad
            print(f"  ext {tid} {ep.upper():3s}: {got:.6g} vs {want:.6g}{'  <-- DRIFT' if bad else ''}")
    # annotation cross-check: npz sig-rate == summary sig-rate
    if ROB is not None and (RESULTS / "robustness_pvalues.npz").exists():
        pv = dict(np.load(RESULTS / "robustness_pvalues.npz"))
        print("[verify] Fig-annotation sig-rates vs robustness_summary.json:")
        mism = 0
        for key, arr in pv.items():
            parts = key.split("_")
            src = parts[0]
            e = (_rob("internal", None, parts[1]) if src == "internal"
                 else _rob("external", parts[1], parts[2]))
            if e is None:
                continue
            got = float(np.mean(np.asarray(arr, float) < 0.05))
            if abs(got - e["sig_rate"]) > 5e-3:
                mism += 1
                print(f"  MISMATCH {key}: npz {got:.3f} vs summary {e['sig_rate']:.3f}")
        print(f"  annotation cross-check: {'OK' if mism == 0 else f'{mism} MISMATCH'}")
        ok &= mism == 0
    print(f"[verify] {'ALL MATCH' if ok else 'PROBLEM — inspect before use'}")
    return ok


# --------------------------------------------------------------------------- #
# figure composition
# --------------------------------------------------------------------------- #

def _km_panel(ax, cmp, title, *, ylabel=True, xlabel=True):
    plots.plot_km_overlay(cmp, ax=ax)
    if (leg := ax.get_legend()) is not None:
        leg.remove()
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel("Survival probability S(t)" if ylabel else "")
    ax.set_xlabel("Time (days)" if xlabel else "")
    return ax


def _km_legend(fig):
    fig.legend(handles=[Line2D([0], [0], color=REAL, lw=2.2, label="real"),
                        Line2D([0], [0], color=SIM, lw=2.2, label="simulated (representative)")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))


def fig1_internal(cmps, sld, n_test):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    _km_panel(axes[0], cmps["pfs"],
              f"(a) PFS — held-out n={n_test}\n{annot('internal', None, 'pfs')}")
    _km_panel(axes[1], cmps["os"], ylabel=False,
              title=f"(b) OS — held-out n={n_test}\n{annot('internal', None, 'os')}")
    if sld is not None:
        plots.plot_trajectory_bands(sld["times"], sld["samples"],
                                    observed_times=sld["obs_t"], observed_sld=sld["obs_sld"],
                                    ax=axes[2], title=f"(c) SLD trajectory — patient {sld['pid']}")
        axes[2].set_xlabel("Time (days)")
    fig.suptitle("Fig. 1  Internal validation on held-out trial-438 patients "
                 "(representative curves; deployed-estimand distribution annotated)",
                 fontsize=11.5, y=1.02)
    _km_legend(fig)
    fig.tight_layout()
    return fig


def fig2_external_km(ext):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
    for col, tid in enumerate(TRIAL_ORDER):
        res, surv = ext[tid]["res"], ext[tid]["surv"]
        head = f"Trial {tid} · {REGIMEN_SHORT[tid]}\n{res.line}, {res.histology_label}, n={res.n_real}"
        _km_panel(axes[0, col], surv["os"], ylabel=(col == 0), xlabel=False,
                  title=f"{head}\nOS: {annot('external', tid, 'os')}")
        if "pfs" in surv:
            _km_panel(axes[1, col], surv["pfs"], ylabel=(col == 0),
                      title=f"PFS: {annot('external', tid, 'pfs')}")
        else:
            ax = axes[1, col]
            ax.text(0.5, 0.5, "PFS not derivable\n(no progression dates)",
                    ha="center", va="center", fontsize=10, color="#555555", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Fig. 2  External survival transport across four independent control arms "
                 "(representative curves; deployed-estimand distribution annotated)",
                 fontsize=11.5, y=1.01)
    _km_legend(fig)
    fig.tight_layout()
    return fig


def fig3_external_bor(ext):
    cats = ["CR", "PR", "SD", "PD"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    for ax, tid in zip(axes.ravel(), TRIAL_ORDER):
        res = ext[tid]["res"]
        rp = [res.bor["real_proportions"].get(c, 0) for c in cats]
        sp = [res.bor["sim_proportions"].get(c, 0) for c in cats]
        x = np.arange(4)
        ax.bar(x - 0.2, rp, 0.4, label="real", color=REAL, edgecolor="white", linewidth=1)
        ax.bar(x + 0.2, sp, 0.4, label="simulated", color=SIM, edgecolor="white", linewidth=1)
        for xs, vs in ((x - 0.2, rp), (x + 0.2, sp)):
            for xi, v in zip(xs, vs):
                ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#222222")
        ax.set_xticks(x); ax.set_xticklabels(cats)
        ax.set_ylabel("Proportion (evaluable)")
        ax.set_ylim(0, max(max(rp), max(sp)) * 1.2 + 0.02)
        ax.set_title(f"Trial {tid} · {REGIMEN_SHORT[tid]} (n={res.n_real})\n"
                     f"{annot('external', tid, 'bor')}", fontsize=9.5)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Fig. 3  External best-overall-response transport (representative proportions; "
                 "deployed-estimand distribution annotated)", fontsize=11.5, y=1.01)
    fig.tight_layout()
    return fig


def fig4_landmark():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    draw_landmark(ax)
    ax.set_title("Fig. 4  Trial 108: subsequent therapy vs OS —\n"
                 "naive split reverses under a 180-day landmark", fontsize=11, color="#222222")
    fig.tight_layout()
    return fig


def fig5_pvalue_distributions():
    """Deployed-estimand p-value distributions: survival panel + BOR panel (log x)."""
    pv = dict(np.load(RESULTS / "robustness_pvalues.npz"))
    surv = [("Internal PFS", "internal_pfs"), ("Internal OS", "internal_os")]
    for tid in TRIAL_ORDER:
        for ep, lab in (("os", "OS"), ("pfs", "PFS")):
            k = f"external_{tid}_{ep}"
            if k in pv:
                surv.append((f"{tid} {lab}", k))
    bor = [(f"{tid} BOR", f"external_{tid}_bor") for tid in TRIAL_ORDER
           if f"external_{tid}_bor" in pv]

    fig, (axS, axB) = plt.subplots(
        2, 1, figsize=(9, 0.44 * (len(surv) + len(bor)) + 2.2),
        gridspec_kw={"height_ratios": [len(surv), len(bor)]})

    def _panel(ax, rows, color, xlo, xticks, xticklabels):
        for i, (name, key) in enumerate(rows):
            p = np.clip(np.asarray(pv[key], float), 1e-95, 1.0)
            p = p[np.isfinite(p)]
            vp = ax.violinplot([np.log10(p)], positions=[i], vert=False, widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.6); b.set_edgecolor(color)
            ax.text(0.55, i, f"{100 * np.mean(p < 0.05):.0f}% sig", va="center", ha="left",
                    fontsize=8.5, color="#222222")
        ax.axvline(np.log10(0.05), color=THRESH, ls="--", lw=1.4)
        ax.set_xlim(xlo, 1.6)
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
        ax.invert_yaxis()
        ax.spines["left"].set_visible(False); ax.tick_params(left=False)

    _panel(axS, surv, REAL, np.log10(3e-5),
           [0, -1.30103, -2, -3, -4], ["1", "0.05", "0.01", "1e-3", "1e-4"])
    axS.text(np.log10(0.05), -0.75, "p=0.05", color=THRESH, fontsize=8.5, ha="center")
    axS.set_title("Fig. 5  Posterior-predictive p-value distributions (deployed estimand, "
                  "1000 replicates)\nSurvival endpoints — transport = mass right of p=0.05",
                  fontsize=10.5)
    _panel(axB, bor, SIM, -92,
           [-1.30103, -20, -40, -60, -80], ["0.05", "1e-20", "1e-40", "1e-60", "1e-80"])
    axB.set_title("Best overall response (χ²) — non-transport = mass left of p=0.05", fontsize=10)
    axB.set_xlabel("p-value (log scale)")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-draws", type=int, default=400)          # representative replicate pool
    p.add_argument("--num-warmup", type=int, default=1000)      # canonical
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny MCMC smoke run (figures WILL differ)")
    args = p.parse_args(argv)
    if args.quick:
        args.num_warmup = args.num_samples = 80
        args.num_chains = 1
        args.n_draws = 150

    setup_style()
    model, train, test = fit_pipeline(args)
    cmps, sld, n_test = internal_comparisons(model, test, args)
    ext = external_all(model, measurable_cohort(load_438().data), args)
    matched = verify(cmps, ext) if not args.quick else False

    written = []
    written += save(fig1_internal(cmps, sld, n_test), "fig1_internal_km")
    written += save(fig2_external_km(ext), "fig2_external_km")
    written += save(fig3_external_bor(ext), "fig3_external_bor")
    written += save(fig4_landmark(), "fig4_landmark_reversal")
    if ROB is not None and (RESULTS / "robustness_pvalues.npz").exists():
        written += save(fig5_pvalue_distributions(), "fig5_pvalue_distributions")

    print("\n[done] wrote submission figures (300 DPI PNG + vector PDF):")
    for w in written:
        print(f"  {w.relative_to(ROOT)}")
    if not args.quick and not matched:
        print("\nWARNING: representative curves drifted or annotations mismatch — inspect before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
