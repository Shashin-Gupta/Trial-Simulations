#!/usr/bin/env python
"""Phase 1 gate: reproduce the committed baseline at n_draws=400.

Fits the model deterministically (seed=0, default MCMC, chain_method='sequential')
and confirms the regenerated internal + external log-rank p and BOR chi-square
reproduce the committed results/real_data/*_validation.json and external/*.json.
Non-destructive: reads the committed JSONs, writes nothing under results/.

    .venv/bin/python scripts/verify_baseline.py

Exit 0 = ALL MATCH (baseline reproduced); exit 1 = drift (do not proceed).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_real_data_validation import measurable_cohort, stratified_split  # noqa: E402
from vca.data_processing.pds_trials import VALIDATION_TRIALS, load_438, load_trial  # noqa: E402
from vca.models.baseline import MarginalResamplingModel  # noqa: E402
from vca.models.tgi_survival import TGISurvivalModel  # noqa: E402
from vca.validation.external import external_validate_trial  # noqa: E402
from vca.validation.pipeline import run_validation  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "real_data"
N_DRAWS = 400
SEED = 0
# Tolerance: the fit is deterministic, so we expect near-bit-exact reproduction.
# Allow a tiny relative slack for float/JSON round-trip only.
RTOL, ATOL = 1e-6, 1e-9


def _match(got, want) -> bool:
    if got is None or want is None:
        return got is None and want is None
    return abs(got - want) <= ATOL + RTOL * abs(want)


def main() -> int:
    td = measurable_cohort(load_438().data)
    (train, test), split_kind = stratified_split(td, test_fraction=0.2, seed=SEED)
    print(f"[fit] _438 modelable n={td.n_patients} train={train.n_patients} "
          f"test={test.n_patients} split={split_kind}")
    assert train.n_patients == 140 and test.n_patients == 35, "split does not reproduce 140/35"

    bayes = TGISurvivalModel(num_warmup=1000, num_samples=1000, num_chains=4, seed=SEED)
    print("[fit] fitting Bayesian TGI+survival on real _438 (deterministic)...")
    bayes.fit(train)

    checks: list[tuple[str, float | None, float | None]] = []

    # --- internal: Bayesian ------------------------------------------------
    rep = run_validation(bayes, train, test, n_draws=N_DRAWS, seed=SEED, out_dir=None,
                         model_name="438_internal_bayesian_tgi_survival")
    cj = json.loads((RES / "438_internal_bayesian_tgi_survival_validation.json").read_text())
    for ep in ("pfs", "os"):
        s, cs = rep.endpoints[ep]["survival"], cj["endpoints"][ep]["survival"]
        checks.append((f"internal-bayes {ep} logrank_p", s["logrank_p"], cs["logrank_p"]))
        checks.append((f"internal-bayes {ep} sim_median_days", s["sim_median_days"], cs["sim_median_days"]))
    for k in ("mean_logrank_p", "mean_brier_ipcw", "mean_calibration_ece", "sld_crps_primary"):
        checks.append((f"internal-bayes {k}", rep.headline[k], cj["headline"][k]))

    # --- internal: resampling baseline -------------------------------------
    base = MarginalResamplingModel(min_donors=6)
    repb = run_validation(base, train, test, n_draws=N_DRAWS, seed=SEED, out_dir=None,
                          model_name="438_internal_baseline_resampling")
    cjb = json.loads((RES / "438_internal_baseline_resampling_validation.json").read_text())
    for ep in ("pfs", "os"):
        checks.append((f"internal-base {ep} logrank_p",
                       repb.endpoints[ep]["survival"]["logrank_p"],
                       cjb["endpoints"][ep]["survival"]["logrank_p"]))

    # --- external: all four trials -----------------------------------------
    donor = measurable_cohort(load_438().data)
    for tid in VALIDATION_TRIALS:
        real = load_trial(tid)
        res, _sim, _surv = external_validate_trial(bayes, real, donor, n_draws=N_DRAWS, seed=SEED)
        ej = json.loads((RES / "external" / f"{tid}_external.json").read_text())
        checks.append((f"ext {tid} BOR chi2 p", res.bor.get("p_value"), ej["bor"].get("p_value")))
        for ep in ("pfs", "os"):
            got = getattr(res, ep)
            want = ej.get(ep)
            g = got["logrank_p"] if got else None
            w = want["logrank_p"] if want else None
            if g is None and w is None:
                continue
            checks.append((f"ext {tid} {ep} logrank_p", g, w))

    # --- report ------------------------------------------------------------
    print("\n[verify] regenerated (n_draws=400, seed=0) vs committed JSON:")
    all_ok = True
    for label, got, want in checks:
        ok = _match(got, want)
        all_ok &= ok
        gs = "None" if got is None else f"{got:.6g}"
        ws = "None" if want is None else f"{want:.6g}"
        print(f"  {'OK ' if ok else 'BAD'} {label:38s} {gs:>12s} vs {ws:<12s}{'' if ok else '  <-- DRIFT'}")
    print(f"\n[verify] {'ALL MATCH — baseline reproduced' if all_ok else 'DRIFT DETECTED — investigate MCMC settings'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
