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

> ⚠ **VALIDITY — multi-study pooling.** Pooling several trials introduces
> between-study heterogeneity (different eras, geographies, assessment
> schedules). The hierarchical model can be extended with a study-level random
> effect; the current v1 pools without it. Report per-study calibration, and
> prefer leave-one-study-out validation when ≥3 studies are available.

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

## 6. Go / no-go checks on real PDS data (before trusting anything)

When real Project Data Sphere NSCLC data are loaded, verify before proceeding:

1. **Coverage.** Are the canonical covariates actually populated? What is the
   missingness of biomarkers, ECOG, stage?
2. **Follow-up and events.** Sufficient PFS/OS events for stable survival
   estimation? Median follow-up vs median survival?
3. **Assessment schedule.** How regular are RECIST assessments? Irregular or
   informative visit timing biases TGI fits.
4. **Between-study heterogeneity** if pooling (§2).
5. **KM sanity vs benchmarks.** Do real KM medians fall within the
   ClinicalTrials.gov historical range (§ `vca.data_processing.clinicaltrials`)?

If these checks fail for NSCLC specifically, reconsider the indication
(colorectal fallback) — this is the point at which to consult before proceeding.

---

## 7. Change log

- **v0.1** — Canonical schema; synthetic generator with known DGP; baseline
  resampling model; hierarchical Bayesian TGI + Weibull-survival joint model;
  validation metric suite; ClinicalTrials.gov benchmark puller. Validated on
  synthetic data (parameter recovery + metric behaviour); **not yet run on real
  PDS data**.
