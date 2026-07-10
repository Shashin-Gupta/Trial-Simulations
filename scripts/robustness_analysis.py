#!/usr/bin/env python
"""Phase 2: posterior-predictive robustness of the transport p-values.

The committed pipeline reports a SINGLE posterior-predictive replicate per
endpoint, and that statistic is Monte-Carlo noisy (e.g. changing the draw pool
from 300 to 400 moved trial-272 PFS from p=0.97 to p=0.20). We replace each single
p-value with a DISTRIBUTION over many independent replicates and report the
decision-relevant summaries.

Two estimands (both reported; DEPLOYED is primary)
--------------------------------------------------
The model is fit once (canonical 1000/1000/4, seed 0, deterministic). For a target
cohort we build a large posterior-predictive pool of size POOL == posterior size
(4 chains x 1000 draws) via chunked ``.simulate()`` -- every draw is an independent
posterior-predictive realisation (fresh posterior sample theta, fresh per-patient
random effects, observation noise, and survival draws). From that pool we form two
kinds of replicate, each a MATCHED-SIZE synthetic cohort (exactly one simulated
trajectory per patient -- never an oversized arm, which would inflate log-rank
power and reject transport artificially):

* **DEPLOYED (primary)** -- each patient draws its OWN theta from the full posterior
  (one pool draw per patient, resampled with replacement). This is how a virtual
  control arm is actually deployed and matches the single-replicate statistic
  (``sample_one_per_patient``) whose distribution we are characterising; the large
  full-posterior pool removes the small-pool artefact. Between-replicate variation
  is the per-patient resample (deployment noise).

* **PER-DRAW (sensitivity)** -- each replicate is one pool draw, i.e. a single
  posterior theta shared across the whole cohort (a textbook posterior-predictive
  check with sharp curves). This is a stricter per-parameter-draw calibration check;
  it detects small systematic offsets the deployed marginal arm averages over, so
  its significance rates are higher. Reported for transparency.

Baseline covariates (including the donor-imputed tumour-burden anchor for the
external trials) are held fixed across replicates; only the posterior-predictive
draw and the fair-censoring draw vary.

Interpretive guardrail
----------------------
p-values are ~Uniform(0,1) under the null, so a WIDE 5-95% interval on a
transporting endpoint (e.g. ~0.05-0.95) is EXPECTED and is NOT evidence of a
problem. The decision-relevant quantities are the significance rate (fraction of
replicates with p<0.05) and whether the qualitative transports/fails conclusion is
stable across replicates.

    .venv/bin/python scripts/robustness_analysis.py                 # 1000 reps
    .venv/bin/python scripts/robustness_analysis.py --n-rep 2000    # scale up
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_real_data_validation import measurable_cohort, stratified_split  # noqa: E402
from vca.data_processing.pds_trials import BOR_LEVELS, VALIDATION_TRIALS, load_438, load_trial  # noqa: E402
from vca.models.tgi_survival import TGISurvivalModel  # noqa: E402
from vca.validation.external import _bor_test, classify_bor, matched_population  # noqa: E402
from vca.validation.pipeline import _aligned_events  # noqa: E402
from vca.validation.survival import compare_survival  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "real_data"
TIMES = np.arange(0.0, 721.0, 30.0)
HORIZON = 540.0
FIT_KW = dict(num_warmup=1000, num_samples=1000, num_chains=4, seed=0)
POOL = 4000            # == posterior size (num_chains*num_samples): full-posterior pool
CHUNK = 500            # simulate() draws per chunk (memory bound)
POOL_SEED0 = 7000      # base seed for the pool's simulate() chunks
RESAMPLE0 = 500000     # base seed for DEPLOYED per-patient resamples
CENS0 = 900000         # base seed for fair-censoring draws (one per replicate)
BOR_CODE = {lab: i for i, lab in enumerate(BOR_LEVELS)}   # CR/PR/SD/PD -> 0..3 (NE/other -> 4)
REGIMEN_SHORT = {"141": "Pac+Carbo+Bev", "272": "Gem+Cis",
                 "133": "Placebo+Docetaxel", "108": "Pac+Carbo"}


def build_pool(model, cov):
    """Chunked simulate() -> latent-event pool (POOL, n_pat) + BOR code matrix."""
    pt, pe, ot, oe, codes = [], [], [], [], []
    done, ci = 0, 0
    while done < POOL:
        k = min(CHUNK, POOL - done)
        r = model.simulate(cov, n_draws=k, times=TIMES, seed=POOL_SEED0 + ci)
        pt.append(r.pfs_time); pe.append(r.pfs_event); ot.append(r.os_time); oe.append(r.os_event)
        traj = r.sld_mean
        cm = np.empty((k, traj.shape[1]), dtype=np.int8)
        for d in range(k):
            for j in range(traj.shape[1]):
                cm[d, j] = BOR_CODE.get(classify_bor(TIMES, traj[d, j, :], horizon_days=HORIZON), 4)
        codes.append(cm)
        done += k; ci += 1
    return {"pfs": (np.concatenate(pt), np.concatenate(pe)),
            "os": (np.concatenate(ot), np.concatenate(oe)),
            "bor_codes": np.concatenate(codes)}   # (POOL, n_pat) int8


def _counts(code_row) -> pd.Series:
    c = np.bincount(code_row[code_row < 4], minlength=len(BOR_LEVELS))[:len(BOR_LEVELS)]
    return pd.Series(c, index=list(BOR_LEVELS))


def survival_dist(pool, ep, rt, re_, *, n_rep, deployed):
    """Per-replicate log-rank p (+stat). deployed=True: per-patient theta; else per-draw."""
    time, event = pool[ep]                          # (POOL, n_pat)
    n_pat = time.shape[1]
    ok = np.isfinite(rt) & np.isfinite(re_)
    m = min(int(ok.sum()), n_pat)
    ps, stats = [], []
    for r in range(n_rep):
        if deployed:
            idx = np.random.default_rng(RESAMPLE0 + r).integers(0, time.shape[0], size=n_pat)
            latent = time[idx, np.arange(n_pat)]
        else:
            latent = time[r]                        # one pool draw = one-theta cohort
        cmp = compare_survival(ep, latent[:m], rt[ok][:m], re_[ok][:m].astype(int), seed=CENS0 + r)
        ps.append(cmp.logrank_p); stats.append(cmp.logrank_stat)
    return np.asarray(ps, float), np.asarray(stats, float)


def bor_dist(pool, real_counts, *, n_rep, deployed):
    codes = pool["bor_codes"]                        # (POOL, n_pat)
    n_pat = codes.shape[1]
    ps, stats = [], []
    for r in range(n_rep):
        if deployed:
            idx = np.random.default_rng(RESAMPLE0 + 7 + r).integers(0, codes.shape[0], size=n_pat)
            row = codes[idx, np.arange(n_pat)]
        else:
            row = codes[r]
        bt = _bor_test(real_counts, _counts(row))
        ps.append(bt.get("p_value")); stats.append(bt.get("statistic"))
    return (np.asarray([x for x in ps if x is not None], float),
            np.asarray([x if x is not None else np.nan for x in stats], float))


def summarize(p, stat):
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    stat = np.asarray(stat, float)
    return {"n_rep": int(p.size),
            "sig_rate": float(np.mean(p < 0.05)) if p.size else float("nan"),
            "median_p": float(np.median(p)) if p.size else float("nan"),
            "p5": float(np.percentile(p, 5)) if p.size else float("nan"),
            "p95": float(np.percentile(p, 95)) if p.size else float("nan"),
            "stat_median": float(np.nanmedian(stat)) if stat.size else float("nan")}


def endpoint_entry(pool, kind, *, n_rep, rt=None, re_=None, real_counts=None):
    """Compute DEPLOYED (primary) + PER-DRAW (sensitivity) summaries for one endpoint."""
    out, arrs = {}, {}
    for deployed, key in [(True, "deployed"), (False, "per_draw")]:
        if kind == "bor":
            p, st = bor_dist(pool, real_counts, n_rep=n_rep, deployed=deployed)
        else:
            p, st = survival_dist(pool, kind, rt, re_, n_rep=n_rep, deployed=deployed)
        out[key] = summarize(p, st)
        if deployed:
            arrs["p"] = p
    return out, arrs["p"]


def _representative_p():
    rep = {}
    ij = json.loads((RES / "438_internal_bayesian_tgi_survival_validation.json").read_text())
    rep["internal"] = {ep: ij["endpoints"][ep]["survival"]["logrank_p"] for ep in ("pfs", "os")}
    rep["external"] = {}
    for tid in VALIDATION_TRIALS:
        ej = json.loads((RES / "external" / f"{tid}_external.json").read_text())
        rep["external"][str(tid)] = {"bor": ej["bor"].get("p_value"),
                                     **{ep: (ej.get(ep) or {}).get("logrank_p") for ep in ("pfs", "os")}}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rep", type=int, default=1000, help="replicates per endpoint (>=500)")
    args = ap.parse_args(argv)
    R = args.n_rep
    assert R <= POOL, f"n-rep {R} exceeds pool size {POOL} (per-draw needs n-rep<=POOL)"

    td = measurable_cohort(load_438().data)
    (train, test), split_kind = stratified_split(td, test_fraction=0.2, seed=0)
    print(f"[fit] canonical {FIT_KW}; split={split_kind} test={test.n_patients}")
    model = TGISurvivalModel(**FIT_KW); model.fit(train)
    donor = measurable_cohort(load_438().data)

    summary = {"n_rep": R, "pool": POOL, "config": FIT_KW,
               "primary_estimand": "deployed",
               "estimands": {"deployed": "per-patient posterior draw (deployed virtual control arm; primary)",
                             "per_draw": "single posterior theta per cohort (stricter posterior-predictive check; sensitivity)"},
               "representative_p_canonical": _representative_p(),
               "internal": {}, "external": {}}
    pv = {}

    print("[internal] 438 held-out PFS/OS ...")
    cov_i = test.covariates()
    ev = _aligned_events(test, list(cov_i.index))
    pool_i = build_pool(model, cov_i)
    for ep in ("pfs", "os"):
        rt = ev[f"{ep}_time_days"].to_numpy(float); re_ = ev[f"{ep}_event"].to_numpy(float)
        entry, parr = endpoint_entry(pool_i, ep, n_rep=R, rt=rt, re_=re_)
        summary["internal"][ep] = entry
        pv[f"internal_{ep}"] = parr
        print(f"  {ep.upper()}: deployed sig={entry['deployed']['sig_rate']:.3f} "
              f"med={entry['deployed']['median_p']:.3f} "
              f"[{entry['deployed']['p5']:.3f},{entry['deployed']['p95']:.3f}] | "
              f"per-draw sig={entry['per_draw']['sig_rate']:.3f}")

    for tid in VALIDATION_TRIALS:
        real = load_trial(tid); cov_e = matched_population(real, donor, seed=0)
        pool_e = build_pool(model, cov_e); evx = real.data.events
        d = {}
        for ep in ("pfs", "os"):
            rt = pd.to_numeric(evx[f"{ep}_time_days"], errors="coerce").to_numpy(float)
            re_ = pd.to_numeric(evx[f"{ep}_event"], errors="coerce").to_numpy(float)
            if np.isfinite(re_).sum() == 0 or np.nansum(re_) == 0:
                continue
            entry, parr = endpoint_entry(pool_e, ep, n_rep=R, rt=rt, re_=re_)
            d[ep] = entry; pv[f"external_{tid}_{ep}"] = parr
        real_bor = real.bor[real.bor.isin(BOR_LEVELS)]
        rc = real_bor.value_counts().reindex(list(BOR_LEVELS), fill_value=0)
        entry, parr = endpoint_entry(pool_e, "bor", n_rep=R, real_counts=rc)
        d["bor"] = entry; pv[f"external_{tid}_bor"] = parr
        summary["external"][str(tid)] = d
        print(f"[external] {tid} ({REGIMEN_SHORT.get(str(tid),'')}):")
        for ep in list(d):
            print(f"    {ep.upper():3s}: deployed sig={d[ep]['deployed']['sig_rate']:.3f} "
                  f"med={d[ep]['deployed']['median_p']:.3g} "
                  f"[{d[ep]['deployed']['p5']:.3g},{d[ep]['deployed']['p95']:.3g}] | "
                  f"per-draw sig={d[ep]['per_draw']['sig_rate']:.3f}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    rows = []
    for ep, e in summary["internal"].items():
        rows.append({"source": "internal", "trial": "438", "endpoint": ep,
                     **{f"deployed_{k}": v for k, v in e["deployed"].items()},
                     **{f"perdraw_{k}": v for k, v in e["per_draw"].items()}})
    for tid, dd in summary["external"].items():
        for ep, e in dd.items():
            rows.append({"source": "external", "trial": tid, "endpoint": ep,
                         **{f"deployed_{k}": v for k, v in e["deployed"].items()},
                         **{f"perdraw_{k}": v for k, v in e["per_draw"].items()}})
    pd.DataFrame(rows).to_csv(RES / "robustness_summary.csv", index=False)
    np.savez_compressed(RES / "robustness_pvalues.npz", **pv)   # DEPLOYED per-replicate p-values
    print(f"\n[done] wrote robustness_summary.json + .csv + robustness_pvalues.npz ({R} reps, pool={POOL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
