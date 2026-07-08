# Virtual Control Arms

**A generative model that simulates realistic, patient-level oncology trial
trajectories — tumour size over time, progression, and survival — for a single,
well-characterised indication (advanced NSCLC), validated against real historical
comparator-arm data.**

> Status: **research — validated on real Project Data Sphere data.** Phases 0–2
> are built; the model is now trained and internally validated on a real NSCLC
> comparator arm (`_438`) and externally validated against four independent real
> trials at the aggregate level (`docs/methodology.md` §6–7). ClinicalTrials.gov
> is integrated as a benchmark sanity check; SEER-based real-world calibration is
> planned future work, **not** yet incorporated (so this is a *trial-data*
> validation, not population-level calibration). The end goals are (1) a
> bioRxiv-publishable validation paper for one indication/endpoint, and (2) a
> lightweight tool that turns a proposed trial design into a simulated
> power/feasibility analysis. See the [roadmap](#roadmap).

---

## What is a "virtual control arm"? (plain language)

Most randomized trials need a **control arm** — patients who receive the standard
of care — to know whether the experimental treatment actually helps. Recruiting a
control arm is slow, expensive, and (for the patients randomized to it) means not
receiving the investigational drug.

A **virtual control arm** replaces or augments those control patients with
*simulated* ones. You take a proposed patient (their age, cancer stage,
performance status, tumour burden, biomarkers) and a model *generates* a
plausible disease trajectory for them: how their tumour would grow or shrink over
time, when their disease would progress, how long they would survive — **as if
they had received standard of care**. The model is learned from thousands of real
control patients in past trials.

This is only trustworthy if the simulated patients behave statistically like real
control patients. **That validation is the actual scientific contribution of this
project** — not the model itself. Enterprise vendors (Unlearn.AI, Certara) do
versions of this for pharma; the wedge here is a rigorous, *open*, honestly
validated methodology for **one** indication, publishable and auditable, before
any product.

## The clinical / statistical problem

Formally, we learn a conditional generative distribution

> p( tumour-size trajectory SLD(t), progression-free survival, overall survival | baseline covariates )

from historical single-/comparator-arm data, and we validate that draws from it
match held-out real patients on:

- **survival curves** (Kaplan–Meier overlay + log-rank),
- **calibration** (do predicted event probabilities match observed frequencies?),
- **prediction-interval coverage**, and
- **proper scoring rules** (IPCW Brier score, CRPS).

### Indication and endpoint (confirmed)

- **Population:** advanced/metastatic NSCLC (Stage IIIB/IV), Phase II/III
  comparator arms from Project Data Sphere.
- **Generative object:** RECIST 1.1 **sum of longest diameters (SLD)** trajectory
  (mm) over time, modelled with a bi-exponential tumour-growth-inhibition (TGI)
  model.
- **Endpoints:** **PFS** and **OS** (Weibull, coupled to tumour growth), plus a
  landmark binary "progression by month 6" for calibration.

NSCLC was chosen over prostate/breast because Project Data Sphere has the most
comparator-arm volume there and its RECIST tumour measurements are clean (prostate
is often bone-predominant and non-measurable by RECIST; breast is strongly
receptor-stratified). Colorectal is the fallback. Full rationale and every
flagged validity concern are in [`docs/methodology.md`](docs/methodology.md).

---

## Quick start

### Environment

The scientific stack (incl. NumPyro/JAX) installs cleanly on **Python 3.11–3.14**.
If JAX has no wheel for your interpreter yet, use **Python 3.11 or 3.12** — the
rest of the pipeline (data layer, baseline model, full validation) runs without
JAX via the dependency-light baseline model.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # core + dev tooling (pytest, ruff, jupytext)
pip install -e ".[bayes]"        # + NumPyro/JAX for the Bayesian model (optional)
pip install -e ".[sas]"          # + pyreadstat, only if reading PDS .sas7bdat files
```

### Run the pipeline on synthetic data (no gated data needed)

```bash
# 1. Pull real aggregate benchmarks from ClinicalTrials.gov (public, no auth):
vca-fetch-benchmarks --max-studies 60

# 2. Fit + validate the whole pipeline on synthetic NSCLC data:
python scripts/run_demo.py            # writes results/ tables + figures
pytest -q                             # run the test suite
```

`scripts/run_demo.py` generates synthetic patients with a *known* data-generating
process, fits both the baseline and the Bayesian model, and produces the full
validation report — the same code path you will later point at real data.

### Run the real-data validation (Project Data Sphere)

Real patient-level data is access-gated and **must never be committed**. Once the
five NSCLC comparator-arm datasets are placed under `data/raw/<trial_id>/`
(see [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md)):

```bash
pip install -e ".[bayes,sas]"                 # NumPyro/JAX + pyreadstat
python scripts/profile_trials.py              # Step 1: per-trial data profiles
python scripts/run_real_data_validation.py    # Steps 2–3: internal + external validation
```

This trains the Bayesian model on `_438`, scores it on the held-out 20%, then
runs the external aggregate validation (matched synthetic population → simulated
BOR + PFS/OS vs each real trial), and finally checks all simulated medians
against the ClinicalTrials.gov benchmark. Outputs land in `results/real_data/`
(git-ignored). Each trial is loaded via `vca.data_processing.pds_trials`:

```python
from vca.data_processing.pds_trials import load_trial
rt = load_trial("272")        # RealTrial: canonical TrialData + aggregate BOR
rt.data.validate(strict=False)
```

> Some sponsors' `.sas7bdat` files use a compression `pyreadstat` cannot read;
> `vca.data_processing.sas.read_sas_any` falls back to pandas automatically.

---

## Project structure

```
.
├── README.md
├── pyproject.toml / requirements.txt   # deps; [bayes] extra = NumPyro/JAX
├── data/
│   ├── raw/           # manually downloaded source data (git-ignored)
│   ├── processed/     # canonical-schema local data (git-ignored)
│   ├── benchmarks/    # ClinicalTrials.gov aggregate medians (regenerable)
│   └── DATA_SOURCES.md# how to obtain PDS / SEER / ClinicalTrials.gov data
├── src/vca/
│   ├── data_processing/  # schema, synthetic data, pds_trials (5 real loaders),
│   │                     #   sas reader, CT.gov API, seer (dormant/future)
│   ├── models/           # TrajectoryModel interface, baseline, Bayesian TGI+survival
│   ├── validation/       # calibration, coverage, Brier, CRPS, KM/log-rank,
│   │                     #   pipeline, external (aggregate validation), profiling
│   ├── viz/              # plots
│   └── product/          # Phase 3 wrapper (STUBBED)
├── notebooks/         # 00_data_profiling, 01_validation_report (jupytext .py)
├── scripts/           # run_demo.py (synthetic), profile_trials.py,
│                      #   run_real_data_validation.py (real PDS Steps 2–3)
├── tests/             # pytest
└── docs/methodology.md# modelling choices + flagged validity concerns (Methods draft)
```

**Note on layout.** The package is a proper `src/`-layout package named `vca`
(so `from vca.models import ...` works after `pip install -e .`). The
`data_processing / models / validation / viz` subpackages match the structure in
the project brief.

## The model interface (swap models without rewriting the pipeline)

Every model implements one small interface (`vca.models.base.TrajectoryModel`):

```python
model.fit(trial_data)                       # trial_data: canonical TrialData
result = model.simulate(covariates,         # one row per virtual patient
                        n_draws=200)        # -> SimulationResult
result.predicted_event_prob("pfs", t=180)   # P(progression by day 180) per patient
km_time, km_event = result.sample_one_per_patient("os")   # a simulated cohort
```

Two implementations ship today, and a future conditional VAE / diffusion model
over trajectories can drop in behind the same interface:

- **`MarginalResamplingModel`** — dependency-light within-stratum Kaplan–Meier
  resampling baseline (the bar the Bayesian model must beat).
- **`TGISurvivalModel`** — hierarchical Bayesian tumour-growth-inhibition +
  Weibull-survival **joint** model (NumPyro/NUTS), the primary scientific model.

## Validation output

`vca.validation.run_validation(model, train, test)` produces a `ValidationReport`
with calibration, PI coverage, IPCW Brier, CRPS, and KM/log-rank results, written
as both a machine-readable table (`results/*.json` / `.csv`) and figures — ready
to cite in a paper draft. Metrics are always reported for the Bayesian model
*and* the baseline.

---

## Roadmap

- [x] **Phase 0** — canonical schema, synthetic data with known DGP, dataset
  loaders (PDS/SEER), live ClinicalTrials.gov benchmark puller.
- [x] **Phase 1** — `TrajectoryModel` interface; resampling baseline; hierarchical
  Bayesian TGI + survival joint model.
- [x] **Phase 2** — held-out validation suite (calibration, coverage, Brier,
  CRPS, KM/log-rank); validated on synthetic data.
- [x] **Phase 2 on real data** — Project Data Sphere NSCLC: trained + internally
  validated on `_438`; externally validated aggregates against four independent
  trials; ClinicalTrials.gov benchmark sanity check passed. Results and honest
  limitations in `docs/methodology.md` §6–9. *Paper write-up in progress.*
- [ ] **Future: SEER real-world calibration** — fold population-level SEER
  survival in as an external calibration layer (the loader exists but is
  dormant); trial populations are healthier/more selected than the general
  population, so this is a named limitation until done.
- [ ] **Phase 3 (stubbed only)** — CLI / Streamlit tool: input a trial design,
  get a simulated power/feasibility analysis. Do **not** build until the
  real-data validation is reviewed.

## Honesty policy

Per the project's priorities, this repository favours a well-calibrated simple
model with an honest validation section over an impressive-looking complex model
with hand-wavy validation. Modelling assumptions that could threaten validity are
flagged **⚠ VALIDITY** in `docs/methodology.md` rather than glossed over.

## Data ethics & licensing

- **Code:** MIT (`LICENSE`).
- **Data:** *not* covered by the code license. Project Data Sphere, SEER, and any
  other source are governed by their own data use agreements, which prohibit
  redistribution. **No patient-level data is included in this repo and none may
  ever be committed** (`.gitignore` enforces this defensively).

## Citation

```bibtex
@software{gupta_virtual_control_arms_2026,
  author  = {Gupta, Shashin},
  title   = {Virtual Control Arms: Generative Patient-Level Simulation for
             Oncology Trials (NSCLC)},
  year    = {2026},
  url     = {https://github.com/<your-org>/virtual-control-arms}
}
```
