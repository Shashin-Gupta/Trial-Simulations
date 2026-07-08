# Methodology

This document records the modelling choices, statistical assumptions, and
validity concerns for the Virtual Control Arms project. It is written to become
the basis of the Methods section of a paper, and it is updated as the code
evolves. Points where an assumption could threaten the validity of an eventual
publication are flagged **⚠ VALIDITY**.

---

## 1. Problem definition

We aim to learn a generative model of patient-level disease trajectories for a
single, well-characterised indication — **advanced/metastatic non-small-cell
lung cancer (NSCLC)** — from historical **comparator-arm** trial data, and to
validate that simulated ("virtual") patients reproduce the statistical behaviour
of real held-out control patients. A validated model supports two uses: (a) a
*virtual control arm* that augments or replaces a concurrent control in a small
trial, and (b) a design tool that returns simulated power/feasibility for a
proposed trial (Phase 3, not yet built).

### Indication and endpoint (confirmed choice)

- **Population.** Advanced/metastatic NSCLC (Stage IIIB/IV), comparator arms of
  Phase II/III trials from Project Data Sphere.
- **Primary generative object.** The longitudinal RECIST 1.1 **sum of longest
  diameters (SLD)** of target lesions, in millimetres, over time.
- **Primary validated endpoints.** **Progression-free survival (PFS)** (time to
  RECIST progression or death) and **overall survival (OS)**. A binary landmark
  endpoint — progression by a fixed time (e.g. 6 months) — is used for
  proper-scoring-rule calibration.
- **Covariates.** age, sex, ECOG performance status, stage, histology
  (squamous / non-squamous), smoking, prior therapy lines, baseline SLD, number
  of target lesions, and biomarkers (EGFR/ALK/PD-L1) *where available*.

**Why NSCLC, and why not another tumour type.** Project Data Sphere's largest
comparator-arm holdings are in NSCLC and colorectal cancer, so both give strong
sample sizes. NSCLC was chosen for the first indication because of data volume
and a dedicated PDS External Control Arm program. Colorectal is a viable
alternative with comparably clean RECIST data. **Prostate and breast are worse
first choices for a RECIST-trajectory model**: metastatic prostate cancer is
frequently bone-predominant and non-measurable by RECIST (PSA kinetics, not SLD,
carry the signal), and metastatic breast cancer is strongly stratified by
receptor subtype. If validation on real PDS data reveals NSCLC-specific problems
(see §6), colorectal is the fallback.

> ⚠ **VALIDITY — biomarker era.** Many older PDS comparator arms predate routine
> EGFR/ALK/PD-L1 testing, so those covariates are missing not at random and the
> treatment/biology differs from contemporary standard of care. A model fit on
> older data may be miscalibrated for a modern trial population. We (a) add
> missing-indicators for biomarkers, (b) restrict claims to the represented
> population, and (c) plan an era-stratified sensitivity analysis.

---

## 2. Data schema and splitting

All sources are mapped to a shared three-table canonical schema (`baseline`,
`longitudinal`, `events`; see `vca.data_processing.schema`) so models never
depend on a sponsor's column names. Time is days from randomization; tumour size
is SLD in mm; event indicators are 1 = event, 0 = right-censored.

**Splitting is on patients, never on measurements.** Holding out later
measurements of a patient whose earlier measurements are in the training set
would leak information and inflate apparent performance
(`TrialData.train_test_split`).

> **DESIGN NOTE — single-trial training, external aggregate validation.** The
> original plan was to *pool* several comparator arms for training. When the real
> Project Data Sphere data arrived, only **one** trial (`_438`, Eli Lilly
> H3E-US-S130) turned out to carry patient-level RECIST lesion trajectories; the
> other four supply best-overall-response (BOR) and PFS/OS only. We therefore
> train and internally validate on `_438` alone, and use the other four for
> **external aggregate validation** (§6–7). This is a legitimate and arguably
> stronger design — it tests whether a model that never saw a trial's patients
> can reproduce that trial's real reported outcomes from a matched synthetic
> population — but it must be stated plainly that lesion-data availability, **not**
> a deliberate methodological choice, is what drove it (a limitation for the paper:
> we cannot pool lesion-level dynamics across trials, and between-study
> heterogeneity is assessed only at the aggregate level).

---

## 3. Model v1 — hierarchical Bayesian TGI + survival joint model

Implemented in NumPyro (`vca.models.tgi_survival.TGISurvivalModel`). The design
deliberately favours an interpretable, well-calibrated model over a flashy one.

### 3.1 Longitudinal sub-model (tumour dynamics)

A Stein bi-exponential tumour-growth-inhibition (TGI) model for each patient *i*:

$$\mathrm{SLD}_i(t) = y_{0,i}\left(e^{-d_i t} + e^{g_i t} - 1\right)$$

with a shrinkage rate $d_i$ and a regrowth rate $g_i$. Observation model is
multiplicative log-normal:
$\log \mathrm{SLD}^{obs}_{ij} \sim \mathrm{Normal}(\log \mathrm{SLD}_i(t_{ij}), \sigma_{obs})$.

Rates are hierarchical and covariate-dependent, using a non-centred
parameterization for sampling efficiency:

$$\log d_i = \mu_d + X_i^\top\beta_d + \sigma_d z^d_i,\quad
  \log g_i = \mu_g + X_i^\top\beta_g + \sigma_g z^g_i,\quad z \sim N(0,1).$$

Baseline SLD is used as the anchor $y_{0,i}$.

> ⚠ **VALIDITY — measurement error in the anchor.** Treating baseline SLD as
> known ignores its measurement error; this can bias early-trajectory
> uncertainty. A latent $y_{0,i}$ with an informative prior is a planned
> refinement.

### 3.2 Time-to-event sub-model (survival), coupled to growth

PFS and OS each follow a Weibull proportional-hazards model whose scale depends
on covariates **and on the shared latent growth rate** — this coupling is what
makes the model *joint* rather than two independent regressions:

$$\text{scale}^E_i = \exp\!\big(a_E + X_i^\top\gamma_E + \theta_E(\log g_i - \mu_g)\big),
  \quad T^E_i \sim \mathrm{Weibull}(\text{scale}^E_i, k_E),\ E\in\{\text{PFS},\text{OS}\}.$$

Right-censoring enters through the Weibull survival function. A negative
$\theta_E$ (faster-growing tumours die/progress sooner) is the expected sign and
is recovered on synthetic data.

> ⚠ **VALIDITY — semi-competing risks.** PFS and OS are modelled as separate
> Weibulls; the model does not enforce OS ≥ PFS. An illness-death /
> semi-competing-risks structure is a planned upgrade. Report the fraction of
> simulated patients violating OS ≥ PFS as a diagnostic.

> ⚠ **VALIDITY — informative censoring.** We assume censoring is independent of
> the event process given covariates. Comparator arms with differential dropout
> (e.g. toxicity-driven) violate this; check censoring patterns per study.

### 3.3 Priors

Weakly informative priors centred on clinically plausible values (shrinkage
~0.004/day, growth ~0.002/day, PFS scale ~150 d, OS scale ~300 d, Weibull shape
~1.2). Priors are documented inline in `_numpyro_model`. Prior predictive checks
should confirm they place mass on realistic SLD and survival ranges.

### 3.4 Simulation (generating a virtual patient)

For a new covariate vector we draw the patient's random effects from the fitted
population distribution, evaluate the TGI trajectory over a time grid (with
observation noise), and draw PFS/OS from the coupled Weibull. Predictive draws
combine posterior parameter uncertainty with outcome noise. Event times are
returned *uncensored* (latent); censoring is applied by validation code only
where a fair comparison to observed data demands it.

---

## 4. Baseline model (the bar to beat)

`vca.models.baseline.MarginalResamplingModel` is a transparent nonparametric
baseline: within coarse covariate strata (age band × ECOG × stage group ×
histology, with backoff to marginal), it resamples donor SLD trajectories and
draws event times by inverting the stratum Kaplan–Meier curve. It needs no
Bayesian machinery and is well-calibrated where strata are well populated, so the
Bayesian model must **beat it on held-out metrics** to justify its complexity.

> ⚠ **VALIDITY — no extrapolation.** The baseline cannot represent covariate
> combinations absent from the data and collapses to coarse strata under
> sparsity. Its role is a calibrated floor, not a deployable model.

---

## 5. Validation (this is the paper)

Held-out, patient-level validation (`vca.validation`), comparing simulated vs
real test patients:

- **Discrimination / survival curves.** Kaplan–Meier overlay of simulated vs
  real PFS/OS with a log-rank test for divergence; median-survival comparison.
  For a fair comparison, simulated latent times are subjected to the test set's
  empirical censoring pattern.
- **Calibration.** Landmark calibration plots (predicted vs observed event
  probability by decile of predicted risk) at clinically relevant times.
- **Prediction-interval coverage.** Do X% posterior predictive intervals for SLD
  and for event times contain the truth ~X% of the time?
- **Proper scoring rules.** Time-dependent **Brier score** with
  inverse-probability-of-censoring weighting (IPCW) for the binary
  progression-by-*t* endpoint; **CRPS** for continuous SLD prediction and for
  time-to-event predictive distributions.
- **Naive-baseline comparison.** Every metric is reported for both the Bayesian
  model and the resampling baseline.

Outputs are written as a notebook (plots) and a machine-readable results table
(JSON/CSV) suitable for direct citation.

> ⚠ **VALIDITY — subgroup sample size.** Calibration within small subgroups
> (e.g. ECOG ≥ 2, squamous, biomarker-positive) can be unstable. Report subgroup
> Ns alongside metrics and avoid over-interpreting sparse cells.

> ⚠ **VALIDITY — IPCW assumptions.** IPCW requires a correctly specified
> censoring model; under heavy or covariate-dependent censoring, estimate the
> censoring distribution with covariates and report sensitivity.

---

## 6. Real data and the revised validation design

### 6.1 Data sources (this phase)

Two real sources are used in this phase, and one planned source is **not** yet
incorporated:

- **Project Data Sphere (PDS)** — five de-identified NSCLC **comparator-arm**
  datasets (PDS shares the control arm only, so each is effectively single-arm).
  All patient-level files are DUA-protected and never committed
  (`vca.data_processing.pds_trials`).
- **ClinicalTrials.gov** — aggregate historical PFS/OS medians for the
  indication, used only as a coarse plausibility backstop
  (`vca.data_processing.clinicaltrials`; 162 arm-level medians).
- **SEER (planned, NOT yet used).** Population-level real-world calibration via
  SEER is a future addition; the SEER loader (`vca.data_processing.seer`) is
  present but **dormant** and never invoked by the pipeline. Consequently this
  phase is a **trial-data validation**, not a demonstration that the model is
  calibrated to the real-world (non-trial) population — see §9.

The five trials (all metastatic NSCLC control arms):

| id | sponsor / study | control regimen | histology | line | n | lesion SLD? |
|----|-----------------|-----------------|-----------|------|---|-------------|
| **438** | Lilly H3E-US-S130 | Paclitaxel+Carbo+Bev | non-squamous | 1L | 179 | **yes (train)** |
| 141 | Lilly JMHD | Paclitaxel+Carbo+Bev | non-squamous | 1L | 467 | no |
| 272 | Lilly SQUIRE | Gemcitabine+Cisplatin | **squamous** | 1L | 549 | no |
| 133 | Sanofi VITAL | Placebo+Docetaxel | non-sq (mixed) | **2L** | 455 | no |
| 108 | Celgene CA031 | Paclitaxel+Carbo | mixed (~42% sq) | 1L | 532 | no |

> ⚠ **VALIDITY — censoring conventions differ across trials.** These are
> apples-to-not-quite-apples comparisons. `_438`/`141` PFS is censored at
> subsequent anti-cancer therapy (Lilly `ttevent` convention); `272` uses the
> ADaM `adtte` PFS/OS derivation (CNSR flag); `133` (SDTM) has **no populated
> progression date**, so its PFS is *not derived at all* (OS only, from
> `DM.RFENDY` + a DEATH disposition); `108` (legacy CRF) OS/PFS are re-derived
> from end-of-study + follow-up death/last-contact/progression day fields. Where
> a convention could bias a comparison it is flagged in the per-trial reads.

> ⚠ **VALIDITY — legacy code maps.** `108` (CA031) histology/stage/response codes
> are not shipped with a codelist; they were inferred from code frequencies and
> cross-checked against the trial's published histology (~43% squamous) and stage
> (~79% stage IV) distributions. `133` best response uses the RECIST 1.0
> "Incomplete Response/Stable Disease" (`IR`) label mapped to SD.

### 6.2 Go / no-go checks — answered

1. **Coverage.** ECOG is **absent** in `_438` and `141` (dropped from the design
   matrix); biomarkers (EGFR/ALK/PD-L1) absent throughout; stage is constant
   (all IV) in `_438`. Covariates that actually vary in training: age, sex,
   histology (non-sq/other only — **no squamous**), smoking, baseline SLD,
   target-lesion count.
2. **Follow-up and events.** OS maturity 69–81% across trials; PFS events
   130–486 where derivable. Adequate for stable KM estimation.
3. **Assessment schedule.** `_438` target-lesion SLD assessed ~every 6 weeks;
   4 spurious pre-randomisation scans dropped; 4 patients without any baseline
   target measurement excluded → **175 modelable** patients.
4. **KM sanity vs benchmarks.** ✅ All simulated PFS/OS medians fall inside the
   ClinicalTrials.gov p5–p95 band (PFS [2.0, 16.8] mo, OS [6.8, 33.9] mo); the
   internal PFS median lands on the benchmark median (4.7 mo). No outliers
   (`results/real_data/benchmark_sanitycheck.csv`).

The indication (NSCLC) is retained; no trigger to fall back to colorectal.

---

## 7. Real-data validation results

Reproduce with `python scripts/run_real_data_validation.py` (seed 0, 600×2 NUTS,
400 predictive draws). Outputs under `results/real_data/` (git-ignored).

### 7.1 Internal validation on `_438` held-out

`_438` measurable cohort (n=175) split 80/20, **stratified by histology**
(train 140 / test 35). The Bayesian TGI+survival model vs the resampling
baseline on the held-out 20%:

| metric (better) | Bayesian | baseline | synthetic (prior) |
|-----------------|----------|----------|-------------------|
| mean log-rank *p* (↑) | **0.586** | 0.258 | 0.564 |
| IPCW Brier (↓) | 0.209 | 0.209 | 0.172 |
| calibration ECE (↓) | 0.139 | 0.105 | 0.057 |
| SLD CRPS mm (↓) | **22.8** | 26.4 | 11.1 |

PFS and OS KM overlays are statistically indistinguishable from real
(log-rank *p* = 0.63 and 0.54; real vs sim medians 165 vs 142 d and 360 vs
330 d). The Bayesian model **beats the baseline** on the headline survival
calibration (log-rank 0.586 vs 0.258) and SLD CRPS, ties on Brier, and is
slightly worse on ECE — consistent with the synthetic-data conclusion that the
joint model's advantage is in survival-curve calibration.

> **Honest degradation vs synthetic.** Brier (0.17→0.21), ECE (0.06→0.14), and
> SLD CRPS (11→23) are all worse on real data than on the clean synthetic DGP.
> This is expected — real trajectories are messier, `n_test`=35 is small so
> metrics are noisy, and the SLD predictive intervals are somewhat over-dispersed
> (see below). We report it rather than hide it.

> ⚠ **VALIDITY — SLD over-dispersion / physiological cap.** The bi-exponential
> regrowth term is unbounded and the log-normal growth-rate random effect
> (σ_g ≈ 1.2 on real data) has a heavy upper tail, so a few long-horizon draws
> would reach absurd sizes. `TGISurvivalModel.simulate` now clips the growth
> exponent (matching the fit-time likelihood) and caps simulated SLD at
> `max_sld_mm = 1000` mm. The *median* trajectory is well-centred (sim 54 mm vs
> observed 48 mm at 6 months), but predictive **intervals are wide** — a real
> calibration limitation, not just a numerical guard.

### 7.2 External aggregate validation (per trial)

For each trial: a matched synthetic population (its real baseline covariates + a
tumour-burden anchor sampled from `_438`, since these trials do not record
baseline SLD) is simulated with the `_438`-trained model, then compared to the
trial's real aggregates. Large *p* = simulated and real are **not** distinguishable
(the desirable result).

| trial | regimen / line / histology | BOR χ² *p* | PFS log-rank *p* (real/sim mo) | OS log-rank *p* (real/sim mo) |
|-------|----------------------------|-----------|-------------------------------|-------------------------------|
| 141 | Pac+Carbo+Bev / 1L / non-sq | 2e-6 | **0.25** (5.6/5.1) | **0.18** (13.4/11.0) |
| 272 | Gem+Cis / 1L / **squamous** | 2e-5 | **0.20** (5.5/4.3) | **0.99** (9.9/8.7) |
| 133 | Placebo+Docetaxel / **2L** | 2e-58 | n/a (not derivable) | **0.87** (11.7/10.2) |
| 108 | Pac+Carbo / 1L / mixed | 3e-13 | **0.20** (5.3/4.5) | **0.018** (11.1/9.3) |

**Survival generalises for three of four trials.** PFS is indistinguishable
wherever derivable; OS is indistinguishable for 141, 272, and 133. It **fails for
108 OS** (*p*=0.018): the simulated OS median (9.3 mo) is ~2 mo below the real
11.1 mo. Note the simulated OS median is *below* real for **all four** trials —
the model essentially predicts an `_438`-like OS (~10–11 mo) for everyone, so it
"generalises" where the real population's OS is near `_438`'s and drifts where it
is higher; with 108's large n and high maturity this systematic ~1–2.5 mo
optimism gap reaches significance. This is transportability of a *marginal*
survival baseline more than of strong covariate effects (unsurprising, since
`_438` had little covariate variation to learn from).

The squamous trial 272 was expected to be the clearest breakdown; instead its
**survival generalises well** (OS *p*=0.99). That is a genuine — mildly
surprising — finding, not a fix.

**BOR does not reproduce on a formal χ² test for any trial.** But the reasons
differ and matter:

- For the **1L** trials (141, 272, 108) the SD and PR proportions are actually
  close (e.g. 141 real SD 46% / sim 49%, real PR 40% / sim 45%). The χ² failure
  is driven by systematic **under-prediction of PD-as-best-response** (sim
  1–4% vs real 12–13%). The model simulates only the *target-lesion* SLD
  trajectory; much real early PD is driven by **new lesions / non-target
  progression**, which the model cannot represent. (With n≈500 the test also has
  the power to flag small but real discrepancies.)
- For the **2L** trial 133 the breakdown is **severe and qualitatively
  different** (*p*=2e-58): the model over-predicts response ~5-fold (sim PR 49%
  vs real 9%) and under-predicts PD (sim 3% vs real 42%). This is the **expected
  failure** — `_438` training was 1L-only, `prior_lines` is constant 0, so the
  model has no way to represent second-line refractory biology. Reporting this
  is the point of the external design, not something to patch.

**Bottom line for the paper.** A model trained on one nonsquamous first-line
control arm reproduces the *survival* of independent nonsquamous, squamous, and
even second-line control arms at the aggregate level (with a small OS optimism
bias), but does **not** reproduce their RECIST response distributions — it misses
non-target/new-lesion progression everywhere and breaks down entirely on
response for the out-of-domain second-line population.

---

## 8. Go / no-go — verdict

NSCLC is retained. The joint model is internally well-calibrated on real data and
beats the baseline on survival; external survival generalisation is a genuine,
if partial, positive result; BOR generalisation and second-line transportability
are honest negatives worth a full paragraph in the paper, not a footnote.

---

## 9. Limitations (paper §Limitations)

- **No real-world (population-level) calibration.** SEER is not yet incorporated;
  all five sources are **clinical-trial** populations, which are healthier and
  more selected (ECOG 0–1, organ-function thresholds) than the general
  metastatic-NSCLC population. Simulated survival should therefore **not** be read
  as a real-world estimate. Folding in SEER as an external calibration layer is
  explicit future work.
- **Single training trial / narrow covariate support.** `_438` is nonsquamous,
  first-line, stage IV, one regimen, no ECOG, no biomarkers. The model cannot
  represent squamous biology, second-line refractoriness, or biomarker strata;
  apparent survival generalisation partly reflects that NSCLC control-arm OS is
  itself similar (~10–13 mo) across these settings.
- **Target-lesion SLD only.** No non-target lesions or new lesions ⇒ systematic
  under-prediction of PD-as-best-response and, likely, of some PFS events.
- **Heterogeneous, partly legacy data.** Four sponsor formats, inferred legacy
  code maps (108), a coarse OS-only derivation and no PFS for 133, and differing
  censoring conventions (§6.1).
- **Aggregate-only external validation.** The four external trials permit only
  distributional comparison, not patient-level calibration/coverage.
- Prior model-level caveats (§3) still apply: baseline-SLD measurement error,
  no enforced OS ≥ PFS, assumed independent censoring, SLD interval
  over-dispersion (§7.1).

---

## 10. Change log

- **v0.2** — Real Project Data Sphere validation. Loaders for five sponsor
  formats (`vca.data_processing.pds_trials`, `.sas`); revised design (train +
  internally validate on `_438`, externally validate aggregates on four trials);
  RECIST BOR classifier + matched-population external pipeline
  (`vca.validation.external`); simulate-time SLD clipping fix; ClinicalTrials.gov
  benchmark sanity check. SEER remains dormant/future work.
- **v0.1** — Canonical schema; synthetic generator with known DGP; baseline
  resampling model; hierarchical Bayesian TGI + Weibull-survival joint model;
  validation metric suite; ClinicalTrials.gov benchmark puller. Validated on
  synthetic data (parameter recovery + metric behaviour); not yet run on real
  PDS data.
