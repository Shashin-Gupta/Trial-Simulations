#!/usr/bin/env python
"""End-to-end demo of the Virtual Control Arms pipeline on synthetic data.

Generates synthetic NSCLC patients with a known data-generating process, fits the
dependency-light baseline and (optionally) the Bayesian TGI-survival model, runs
the held-out validation suite, and writes a citable metrics table + figures to
``results/``. This is the exact code path you later point at real Project Data
Sphere data — only the data source changes.

    python scripts/run_demo.py                 # baseline + Bayesian (small MCMC)
    python scripts/run_demo.py --no-bayes      # baseline only (no NumPyro needed)
    python scripts/run_demo.py --n-patients 800 --num-samples 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

from vca.data_processing.synthetic import make_synthetic_nsclc
from vca.models.baseline import MarginalResamplingModel
from vca.validation.pipeline import compare_models


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-patients", type=int, default=500)
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--n-draws", type=int, default=300)
    p.add_argument("--num-warmup", type=int, default=400)
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bayes", action="store_true", help="skip the NumPyro model")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    print(f"Generating {args.n_patients} synthetic NSCLC patients (seed={args.seed})...")
    td = make_synthetic_nsclc(args.n_patients, seed=args.seed)
    train, test = td.train_test_split(args.test_fraction, seed=args.seed)
    print(f"  train={train.n_patients}  test={test.n_patients}")

    models = {"baseline_resampling": MarginalResamplingModel(min_donors=8)}
    if not args.no_bayes:
        try:
            from vca.models.tgi_survival import TGISurvivalModel

            models["bayesian_tgi_survival"] = TGISurvivalModel(
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=1,
                seed=args.seed,
            )
        except ImportError as exc:
            print(f"  [skip] Bayesian model unavailable: {exc}")

    print(f"Validating models: {list(models)}")
    headline = compare_models(
        models, train, test,
        n_draws=args.n_draws, seed=args.seed,
        out_dir=args.out, make_figures=True,
    )

    print("\n=== Headline comparison (higher log-rank p / lower Brier,ECE,CRPS = better) ===")
    print(headline.to_string(index=False))
    print(f"\nWrote metrics tables + figures to {Path(args.out).resolve()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
