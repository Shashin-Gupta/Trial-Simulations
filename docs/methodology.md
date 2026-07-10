# Methodology

This document specifies the modelling choices, validation design, and results for
the Virtual Control Arms project: a generative model of patient-level disease
trajectories for advanced non-small-cell lung cancer (NSCLC), learned from
historical comparator-arm trial data, whose simulated ("virtual") control
patients are validated against real held-out and independent-trial outcomes. The
validation — not the model — is the scientific contribution, so the emphasis
throughout is on stating evidence and its limits plainly.

---

## 1. Design

### 1.1 Indication and endpoints

- **Population.** Advanced/metastatic NSCLC (Stage IIIB/IV), comparator (control)
  arms of Phase II/III trials.
- **Primary generative object.** The longitudinal RECIST 1.1 **sum of longest
  diameters (SLD)** of target lesions, in millimetres, over time.
- **Validated endpoints.** **Progression-free survival (PFS)** and **overall
  survival (OS)**; a landmark binary "progression by 6 months" for calibration;
  and, for the external trials, aggregate **best overall response (BOR;
  CR/PR/SD/PD)**.
- **Covariates.** age, sex, ECOG performance status, stage, histology
  (squamous / non-squamous), smoking, prior therapy lines, baseline SLD, number
  of target lesions, and biomarkers (EGFR/ALK/PD-L1) where available.

NSCLC was chosen because Project Data Sphere's comparator-arm holdings and clean
RECIST measurements are strongest there (prostate is frequently bone-predominant
and non-measurable by RECIST; breast is strongly receptor-stratified). Colorectal
was the pre-specified fallback; nothing in the results triggered it.

### 1.2 Data sources and cohorts

Two real sources are used; a third (SEER) is planned but **not** incorporated.

- **Project Data Sphere (PDS)** — five de-identified NSCLC **control-arm**
  datasets. PDS shares the comparator arm only, so each is effectively single-arm.
  All patient-level files are DUA-protected and never committed
  (`vca.data_processing.pds_trials`).
- **ClinicalTrials.gov** — 162 arm-level historical PFS/OS medians for the
  indication, used only as a plausibility backstop
  (`vca.data_processing.clinicaltrials`).
- **SEER (planned, not used here).** Population-level real-world calibration is a
  future addition; the loader (`vca.data_processing.seer`) is present but dormant
  and never invoked. This phase is therefore a **trial-data validation**, not a
  demonstration of real-world (non-trial) calibration.

| id | study | control regimen | histology | line | n | lesion SLD? |
|----|-------|-----------------|-----------|------|---|-------------|
| **438** | Lilly H3E-US-S130 | Paclitaxel+Carbo+Bev | non-squamous | 1L | 179 | **yes (train)** |
| 141 | Lilly JMHD | Paclitaxel+Carbo+Bev | non-squamous | 1L | 467 | no |
| 272 | Lilly SQUIRE | Gemcitabine+Cisplatin | squamous | 1L | 549 | no |
| 133 | Sanofi VITAL | Placebo+Docetaxel | non-sq (mixed) | 2L | 455 | no |
| 108 | Celgene CA031 | Paclitaxel+Carbo | mixed (~42% sq) | 1L | 532 | no |

**Data caveats (reviewer-relevant).** Censoring conventions differ across trials:
`_438`/`141` PFS is censored at subsequent anti-cancer therapy; `272` uses the
ADaM `adtte` derivation; `133` (SDTM) has no populated progression date, so PFS is
**not derived** for it (OS only, from `DM.RFENDY` + a DEATH disposition); `108`
(legacy CRF) OS/PFS are re-derived from end-of-study and follow-up death/last-
contact/progression fields. `108`'s histology/stage/response code lists were not
shipped and were inferred from code frequencies and cross-checked against the
trial's published ~43% squamous / ~79% stage-IV distributions.

### 1.3 Single-trial training, multi-trial external validation

Of the five datasets, **only `_438` carries patient-level RECIST lesion
trajectories**; the other four supply BOR and PFS/OS only. We therefore train and
internally validate on `_438` alone, and use the other four for **external
aggregate validation**. We state plainly that this was **data-driven — the
availability of lesion-level SLD, not a methodological preference**. It is a
legitimate and arguably demanding design: it tests whether a model that never saw
a trial's patients can reproduce that trial's real reported outcomes from a
matched synthetic population. Its cost is that between-trial heterogeneity can be
assessed only at the aggregate level and, more consequentially, that the model's
covariate support is narrow (§4.2).

### 1.4 Generative model

The primary model (`vca.models.tgi_survival.TGISurvivalModel`, NumPyro/NUTS) is a
hierarchical Bayesian joint model in the pharmacometric TGI–OS tradition.

**Longitudinal sub-model (Stein bi-exponential TGI):**

$$\mathrm{SLD}_i(t) = y_{0,i}\left(e^{-d_i t} + e^{g_i t} - 1\right),\qquad
  \log \mathrm{SLD}^{obs}_{ij} \sim \mathrm{Normal}(\log \mathrm{SLD}_i(t_{ij}), \sigma_{obs}),$$

with baseline SLD as the anchor $y_{0,i}$ and hierarchical, covariate-dependent
shrinkage/regrowth rates (non-centred):
$\log d_i = \mu_d + X_i^\top\beta_d + \sigma_d z^d_i$,
$\log g_i = \mu_g + X_i^\top\beta_g + \sigma_g z^g_i$.

**Time-to-event sub-model (Weibull PH, coupled to growth):**

$$\text{scale}^E_i = \exp\!\big(a_E + X_i^\top\gamma_E + \theta_E(\log g_i - \mu_g)\big),\quad
  T^E_i \sim \mathrm{Weibull}(\text{scale}^E_i, k_E),\ E\in\{\text{PFS},\text{OS}\}.$$

Sharing $\log g_i$ across the two likelihoods is what makes the model *joint*. A
negative $\theta_E$ (faster-growing tumours progress/die sooner) is the expected
sign and is recovered on real `_438` data.

**Baseline model (the bar to beat).** `MarginalResamplingModel` resamples donor
SLD trajectories within coarse covariate strata and inverts stratum
Kaplan–Meier curves for event times. The Bayesian model must beat it on held-out
metrics to justify its complexity.

**Simulation.** For a covariate vector we draw the patient's random effects from
the fitted population, evaluate the TGI trajectory (with observation noise), and
draw PFS/OS from the coupled Weibull. The regrowth term is unbounded and the
log-normal growth-rate random effect has a heavy upper tail, so the growth
exponent is clipped (as in the fit-time likelihood) and simulated SLD is capped at
a physiological maximum (`max_sld_mm = 1000`); this prevents a few long-horizon
draws from overflowing while leaving the trajectory median unaffected.

**Modelling assumptions (refinement targets, §5).** Baseline SLD is treated as a
known anchor (ignoring its measurement error); PFS and OS are separate Weibulls
and OS ≥ PFS is not enforced; censoring is assumed independent of the event
process given covariates; the growth→survival link is linear in $\log g_i$.

### 1.5 Validation metrics

**Internal (patient-level, `_438` held-out).** Kaplan–Meier overlay of simulated
vs real PFS/OS with a **log-rank** test and median comparison (simulated latent
times are subjected to the test set's empirical censoring for a fair comparison);
**landmark calibration** (expected calibration error, ECE); time-dependent
**IPCW Brier score**; predictive-**interval coverage**; and **CRPS** for SLD.
Splitting is on patients, never on measurements.

**External (aggregate).** For each trial we build a **matched synthetic
population** — the trial's real baseline covariates, with a tumour-burden anchor
(baseline SLD, target-lesion count) resampled from `_438` because the external
trials do not record it — simulate with the `_438`-trained model, and compare
aggregates to the real trial. **BOR** is classified by a RECIST best-response rule
on the model's noise-free expected trajectory (per-visit CR/PR/SD/PD with
precedence CR > PR > SD > PD; assessment stops at the first progression),
simplified in that it uses target-lesion SLD only, with no confirmation
requirement — see §4.1. BOR distributions are compared by **χ² (Fisher's exact
when sparse)** over evaluable CR/PR/SD/PD; PFS/OS curves by **log-rank** with the
simulated latent times censoring-matched to the real cohort. A large p-value means
simulated and real are not detectably different — the desirable outcome.

---

## 2. Internal validation (`_438` held-out)

The `_438` measurable-disease cohort (n = 175; four patients without a baseline
target-lesion measurement excluded) was split 80/20 **stratified by histology**
(train 140 / test 35). The Bayesian model was scored against the held-out 20%
alongside the resampling baseline; the rightmost column is the prior
synthetic-data result for the same model.

| metric (better) | Bayesian | baseline | synthetic (prior) |
|-----------------|----------|----------|-------------------|
| mean log-rank *p* (↑) | **0.348** | 0.258 | 0.564 |
| IPCW Brier (↓) | **0.207** | 0.209 | 0.172 |
| calibration ECE (↓) | 0.150 | **0.105** | 0.057 |
| SLD CRPS, mm (↓) | **22.7** | 26.4 | 11.1 |

The canonical fit is 1000/1000/4 NUTS, seed 0 (worst R̂ = 1.004, minimum ESS 919);
it is deterministic (`chain_method="sequential"`) and reproduces bit-exactly.
Simulated PFS and OS KM curves are indistinguishable from real across replicates
(deployed significance rate 3% and 9%; median log-rank *p* 0.48 and 0.33;
representative replicate *p* = 0.22 and 0.47, real vs simulated medians 165 vs 173 d
and 360 vs 300 d). The Bayesian model beats the baseline on survival-curve
calibration (mean log-rank 0.348 vs 0.258) and SLD CRPS, ~ties on Brier, and is
worse on ECE — the same pattern as on synthetic data, where the joint model's
advantage was also concentrated in survival calibration.

**Honest degradation vs synthetic.** Brier (0.17 → 0.21), ECE (0.06 → 0.15), and
SLD CRPS (11 → 23) are all worse on real data. This is expected: real trajectories
are messier, `n_test` = 35 makes metrics noisy, and the SLD predictive intervals
are over-dispersed — the trajectory *median* tracks the observed measurements but
the intervals are wide, a real calibration limit rather than a numerical artefact.

**Benchmark plausibility.** Every simulated PFS/OS median (internal and external)
falls inside the ClinicalTrials.gov historical p5–p95 band (PFS [2.0, 16.8] mo,
OS [6.8, 33.9] mo).

---

## 3. External validation (four trials)

Matched synthetic populations were simulated with the `_438`-trained model and
compared to each trial's real aggregates. Because a single simulated cohort gives
one Monte-Carlo-noisy replicate of each transport statistic (changing the draw pool
from 300 to 400 moved trial-272 PFS from *p* = 0.97 to 0.20), we characterise the
distribution over 1000 posterior-predictive replicates and report the significance
rate (fraction with *p* < 0.05). Two estimands are computed, both on matched-size
cohorts (one trajectory per patient, never oversized). The **deployed** estimand
(primary) draws each synthetic patient's parameters independently from the full
posterior — how a virtual control arm is used — with a full-posterior pool removing
the draw-pool artefact; the **per-draw** estimand (sensitivity) shares one posterior
draw across the cohort and is stricter. Table cells give the deployed significance
rate and median *p*, with the representative single replicate (canonical
*n_draws* = 400) and real/sim OS medians in parentheses.

| trial | regimen / line / histology | BOR (deployed) | PFS (deployed) | OS (deployed) |
|-------|----------------------------|----------------|----------------|---------------|
| 141 | Pac+Carbo+Bev / 1L / non-sq | 100% sig (rep 1.5×10⁻⁵) | 1% sig; med 0.61 (rep 0.31; 5.6/5.1) | 6% sig; med 0.40 (rep 0.13; 13.4/11.1) |
| 272 | Gem+Cis / 1L / squamous | 100% sig (rep 4.1×10⁻⁵) | 3% sig; med 0.52 (rep 0.63; 5.5/4.8) | 1% sig; med 0.60 (rep 0.82; 9.9/9.1) |
| 133 | Placebo+Docetaxel / 2L | 100% sig (rep 4.0×10⁻⁵⁹) | n/a (not derivable) | 1% sig; med 0.62 (rep 0.62; 11.7/11.3) |
| 108 | Pac+Carbo / 1L / mixed | 100% sig (rep 7.3×10⁻¹³) | 4% sig; med 0.41 (rep 0.55; 5.3/4.8) | 21% sig; med 0.16 (rep 0.09; 11.1/9.5) |

**Survival transports for all four trials under the deployed estimand.** OS is
significant in only 0.6–6% of replicates for 141, 272 and 133, and 21% for 108 (the
weakest), against a ~5% null; PFS in 1–4% wherever derivable. The squamous trial
272 — a plausible a priori breakdown case, since `_438` is nonsquamous-only —
transports well (OS 1% sig). Under the stricter **per-draw** estimand the three
higher-OS trials become significant in the majority of replicates (OS 73% for 272,
77% for 133, 54% for 108), reflecting the small systematic OS under-prediction
(§4.2); internal validation and trial 141 stay clean.

**BOR does not transport for any trial** — significant in 100% of replicates for all
four. The reasons differ by trial and are mechanistically explained in §4.1.

---

## 4. Two identified limitations, with mechanisms

### 4.1 BOR does not transport — target-lesion-only simulation

For the three first-line trials the SD and PR proportions are actually close
(e.g. 141: real SD 46% / PR 40% vs simulated SD 50% / PR 44%). The χ² failure is
driven by systematic **under-prediction of PD-as-best-response** (simulated
1–5% vs real 12–13%). **Mechanism:** the model simulates only the *target-lesion*
SLD trajectory, whereas much real early progression is driven by **new lesions or
non-target progression**, which the model cannot produce. (With n ≈ 500 the test
also has power to flag small but real discrepancies.)

The second-line trial 133 fails differently and severely (representative χ² *p* =
4×10⁻⁵⁹): simulated response is ~5× too high (PR 52% vs real 9%) and PD is
under-predicted (4% vs 42%). This is the **expected out-of-domain failure** — `_438` training was
first-line only and `prior_lines` is constant, so the model has no representation
of second-line refractory biology. It is consistent with, not separate from, the
target-lesion mechanism: the model applies first-line shrinkage dynamics to a
refractory population it cannot characterise.

**Fix:** model non-target and new lesions (and/or add a confirmed-response rule)
so that simulated progression is not target-SLD-only (§5).

### 4.2 OS compresses toward the training-trial marginal — single-trial covariate limitation

Simulated OS runs below real OS for all four trials (gap 0.3–2.3 mo).
**Mechanism:** with a single training trial offering little covariate variation
(one regimen, stage IV only, nonsquamous only, no ECOG), the survival sub-model
has no basis on which to shift the OS *level* between populations, so it regresses
every cohort toward `_438`'s marginal OS (~10 mo) and under-predicts
longer-lived populations.

Three pieces of evidence support this and, together, rule out the leading
alternatives:

- **The gap directionally tracks the trial's real OS level.** Spearman ρ(gap,
  real-OS median) = **+0.40**: the model under-predicts the higher-OS trials (141,
  108) most. With only four trials this is not statistically resolved (*p* = 0.60)
  and one high-OS trial (133) is a partial exception, but the gap is present and
  non-negative in every trial.
- **The under-prediction is already present internally** — on `_438`'s own
  held-out set, simulated OS median ~9.9 mo vs ~11.8 mo real (§2). A gap present
  within the training trial cannot be a cross-trial transport effect.
- **108 is the weakest survival transport, not a unique failure.** Its OS is
  significant in 21% of deployed replicates (majority under the stricter per-draw
  check) — the higher-OS, high-event-maturity trial in which the systematic
  under-prediction is most readily detected.

**Post-progression subsequent therapy was tested and ruled out** as the cause.
Subsequent-therapy rates rise across the externals (25% → 55%), but there is **no
positive dose-response** with the OS gap (ρ(gap, therapy rate) = **−0.40**, the
wrong sign): the largest-gap trial (141) has the *lowest* rate and is `_438`'s own
regimen. The apparent
within-trial signal — subsequent-therapy patients appearing to live 7–9 months
longer — is **immortal-time bias**: patients must survive to progression and be
fit enough to be re-treated, so the untreated arm is enriched for early deaths. A
180-day **landmark** on 108 (restricting to patients alive at 180 d, flagging
therapy started by then) **reverses** the association (12.3 vs 17.7 mo), confirming
that subsequent therapy marks prognosis rather than driving the model's error.

**Fix:** additional lesion-level training trials / covariate diversity, or a
study-level effect, to give the survival sub-model a transportable basis (§5).

---

## 5. Future work

- **Broaden training data (OS-compression fix).** The single-trial limitation in
  §4.2 is the priority. Additional lesion-level control arms spanning squamous,
  second-line, other regimens, and ECOG variation — or a study-level random effect
  — would give the survival sub-model a basis to move the OS level between
  populations rather than regressing toward one trial's marginal.
- **Model non-target and new lesions (BOR-transport fix).** Adding non-target
  progression and new-lesion appearance (and a RECIST confirmation rule) would let
  simulated progression capture the early PD the target-lesion-only model in §4.1
  misses.
- **SEER-based real-world calibration (separate addition).** Trial populations are
  healthier and more selected than the general metastatic-NSCLC population, so the
  current results are a trial-data validation only. Folding in SEER survival as an
  external calibration layer is planned but independent of the two fixes above.
- **Model refinements.** Semi-competing-risks structure to enforce OS ≥ PFS; a
  latent baseline-SLD anchor to propagate its measurement error; and alternative
  tumour-dynamic summaries (e.g. week-8 tumour ratio) as a sensitivity analysis.

A note on strength of evidence: the OS-compression account in §4.2 is
supported by the available data (the ρ = +0.40 level-tracking, not resolved at four
trials; the internal gap; and the subsequent-therapy exclusion), but establishing it
causally requires a second lesion-level training trial — which is exactly the first
future-work item.

---

## Appendix — change log

- **v0.4** — Canonical rebase + robustness. Pinned the deterministic canonical fit
  (1000/1000/4 NUTS, seed 0; R̂ 1.004, ESS 919; reproduces bit-exactly) after the
  original run's MCMC settings proved unrecorded; regenerated all baseline artifacts
  from it. Added `scripts/robustness_analysis.py`: each transport statistic is now a
  distribution over 1000 posterior-predictive replicates, reported by significance
  rate under two estimands (deployed = primary, per-draw = sensitivity). Headline
  refined: under the deployed estimand survival transports for all four external
  trials (108 OS weakest), BOR fails all; the per-draw sensitivity flags the three
  higher-OS trials. `scripts/export_submission_figures.py` regenerates Figs 1–5
  (300 DPI + vector) with distributional annotations; Fig 5 shows the p-value
  distributions.
- **v0.3** — Consolidated, paper-ready restructure (Design / Internal / External /
  Limitations / Future Work).
- **v0.2** — Real Project Data Sphere validation: five-sponsor loaders; revised
  single-trial-training / external-aggregate design; RECIST BOR classifier +
  matched-population external pipeline; simulate-time SLD clipping fix;
  ClinicalTrials.gov benchmark check; OS under-prediction diagnostic (subsequent
  therapy tested and rejected via dose-response + immortal-time-bias landmark).
- **v0.1** — Canonical schema; synthetic generator with known DGP; baseline
  resampling model; hierarchical Bayesian TGI + Weibull-survival joint model;
  validation metric suite; ClinicalTrials.gov benchmark puller. Validated on
  synthetic data only.
