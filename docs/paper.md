# Transportable survival, non-transportable response: a mechanistic validation of a generative virtual control arm for advanced NSCLC

*Working manuscript draft. Abstract and Results are drafted; Introduction,
Methods, and Discussion are placeholders pending review. All quantities are taken
from `docs/methodology.md`, which is the factual source of truth.*

---

## Abstract

Virtual control arms—simulated patients that substitute for a trial's concurrent
control group—could reduce control-arm enrollment, but their value depends on
whether simulated patients behave like real ones. We report a
transportability-focused validation of a generative virtual control arm for
advanced non-small-cell lung cancer (NSCLC). A hierarchical Bayesian model jointly
representing RECIST tumour-size dynamics (bi-exponential tumour-growth inhibition)
and Weibull progression-free and overall survival (PFS, OS) was trained on a
single first-line control arm—the only one of five public datasets releasing
patient-level longitudinal lesion measurements—and evaluated on held-out patients
and on four independent control arms (spanning squamous histology, second-line
therapy, and alternative regimens) reporting only aggregate response and survival.

Internally, simulated PFS and OS were statistically indistinguishable from
held-out real patients (log-rank *p* = 0.63 and 0.54), and the joint model
outperformed a resampling baseline on survival-curve calibration. Externally, the
results split cleanly by endpoint. Survival transported to three of four
independent trials (OS log-rank *p* = 0.18–0.99), including a squamous trial the
non-squamous-trained model had never seen, and failed for one (*p* = 0.018).
RECIST best-overall-response transported to none (χ² *p* ≤ 2×10⁻⁵). Both failures
had identifiable mechanisms. Response non-transport arises because the model
simulates target-lesion size only, missing the non-target and new-lesion
progression that drives much real early progression. OS was under-predicted by
1–2 months because single-trial training gives the survival sub-model no covariate
basis to shift OS between populations, regressing predictions toward the training
marginal (gap versus real OS level: Spearman ρ = +0.80); a landmark analysis
correcting for immortal-time bias excluded subsequent therapy. Each negative
result carries an identified mechanism, and each mechanism a specific fix.

---

## 1. Introduction

*[Placeholder — to be drafted after Abstract/Results review.]*

---

## 2. Methods

*[Placeholder — to be condensed from `docs/methodology.md` §1 (design, data
sources and cohorts, generative model, validation metrics).]*

---

## 3. Results

We trained the joint tumour-growth-inhibition (TGI) and survival model on the one
control arm that released patient-level lesion trajectories (trial 438, a
first-line non-squamous NSCLC arm; the other four datasets released aggregate
response and survival only). We then asked two questions in sequence: whether the
model's simulated patients reproduce real patients held out from that same trial
(internal validation, §3.1), and whether—without ever seeing their patients—the
model reproduces the aggregate outcomes of four independent control arms (external
validation, §3.2). The results separate cleanly by endpoint: survival transports,
RECIST response does not, and each outcome has a distinct, identifiable cause
(§3.3, §3.4).

### 3.1 Internal validation on held-out patients

Of the 179 patients in the training arm, 175 had a measurable baseline target
lesion and were retained; these were split 80/20, stratified by histology, into
140 training and 35 held-out patients. On the held-out set, simulated PFS and OS
Kaplan–Meier curves were statistically indistinguishable from the real curves
(log-rank *p* = 0.63 and 0.54; real versus simulated median PFS 165 versus 142
days, median OS 360 versus 330 days; Fig. 1). Against the resampling baseline, the
joint model was better calibrated on the survival curves (mean log-rank *p* 0.586
versus 0.258) and on tumour-size prediction (SLD CRPS 22.8 versus 26.4 mm), tied
on the inverse-probability-of-censoring-weighted (IPCW) Brier score (0.209), and
slightly worse on landmark calibration error (expected calibration error, ECE,
0.139 versus 0.105; Table 1). This is the same profile observed previously on
synthetic data with a known data-generating process, where the joint model's
advantage was likewise concentrated in survival-curve calibration.

Relative to that synthetic benchmark, every metric degraded on real data (Brier
0.17→0.21, ECE 0.06→0.14, SLD CRPS 11→23 mm; Table 1). We attribute this to
genuinely messier trajectories, the small held-out sample (n = 35), and
over-dispersed SLD predictive intervals: the simulated trajectory median was well
centred (54 mm versus an observed 48 mm at six months) but its intervals were
wide—a real calibration limitation rather than a numerical artefact. As an
external plausibility check, every simulated PFS and OS median, internal and
external, fell within the ClinicalTrials.gov historical range for the indication
(PFS 2.0–16.8 months, OS 6.8–33.9 months), and the internal simulated PFS median
(4.7 months) equalled the historical median.

**Table 1.** Internal validation on held-out training-arm patients (n = 35).
Bayesian joint model versus the resampling baseline; the final column is the same
Bayesian model's prior result on synthetic data. Arrows indicate the better
direction.

| metric | Bayesian | baseline | synthetic (prior) |
|--------|----------|----------|-------------------|
| mean log-rank *p* (↑) | **0.586** | 0.258 | 0.564 |
| IPCW Brier (↓) | 0.209 | 0.209 | 0.172 |
| calibration ECE (↓) | 0.139 | 0.105 | 0.057 |
| SLD CRPS, mm (↓) | **22.8** | 26.4 | 11.1 |

### 3.2 Survival transports to independent trials; response classification does not

We then tested transportability directly. For each of the four independent
control arms we constructed a synthetic population matched to that trial's real
baseline covariates (with a tumour-burden anchor resampled from the training arm,
since the external trials do not record baseline SLD), simulated it with the model
trained only on trial 438, and compared the simulated aggregates to the trial's
real outcomes. Because these trials report only aggregates, the comparison is
distributional—χ² for best-overall-response (BOR) and log-rank for PFS/OS—where a
large *p*-value indicates that simulated and real are not detectably different.

The results separated cleanly by endpoint (Table 2). **Survival transported to
three of the four trials.** OS was statistically indistinguishable from real for
trials 141, 272, and 133 (log-rank *p* = 0.18, 0.99, and 0.87), and PFS was
indistinguishable wherever it could be derived (*p* = 0.20–0.25; PFS was not
derivable for trial 133, whose data export contains no progression dates).
Transport held even where the training arm's composition should have made it
hardest: trial 272 enrolled exclusively squamous patients and used a different
regimen (gemcitabine–cisplatin), yet its simulated OS matched the real curve
almost exactly (*p* = 0.99; Fig. 2). OS transport failed for a single trial (108,
*p* = 0.018). **RECIST best-overall-response, by contrast, did not transport for
any trial** (χ² *p* ≤ 2×10⁻⁵; Fig. 3). The following two sections dissect each
failure; both have an identifiable mechanism rather than an unexplained gap.

**Table 2.** External aggregate validation. For each independent control arm,
simulated versus real best-overall-response (BOR; χ²) and PFS/OS Kaplan–Meier
(log-rank). Large *p* indicates simulated and real aggregates are not detectably
different. Medians are real / simulated, in months.

| trial | *n* | regimen / line / histology | BOR χ² *p* | PFS log-rank *p* (real/sim) | OS log-rank *p* (real/sim) |
|-------|-----|----------------------------|-----------|-----------------------------|----------------------------|
| 141 | 467 | Pac+Carbo+Bev / 1L / non-sq | 2×10⁻⁶ | 0.25 (5.6 / 5.1) | 0.18 (13.4 / 11.0) |
| 272 | 549 | Gem+Cis / 1L / squamous | 2×10⁻⁵ | 0.20 (5.5 / 4.3) | 0.99 (9.9 / 8.7) |
| 133 | 455 | Placebo+Docetaxel / 2L | 2×10⁻⁵⁸ | n/a (not derivable) | 0.87 (11.7 / 10.2) |
| 108 | 532 | Pac+Carbo / 1L / mixed | 3×10⁻¹³ | 0.20 (5.3 / 4.5) | 0.018 (11.1 / 9.3) |

### 3.3 Response non-transport reflects target-lesion-only simulation

The χ² test rejected agreement between simulated and real BOR distributions in
every trial (Table 2), but the failure was structured, not uniform, and its
structure is diagnostic. In the three first-line trials the stable-disease and
partial-response proportions were in fact close—for trial 141, real SD/PR of
46%/40% versus simulated 49%/45%—and the rejection was driven almost entirely by
systematic under-prediction of progressive disease as best response (simulated
1–4% versus real 12–13%). This is the expected consequence of the model's
generative scope. The model simulates only the target-lesion SLD trajectory,
whereas a substantial fraction of real early progression is defined by new lesions
or unequivocal non-target progression—events the model has no mechanism to
produce. (With roughly 500 patients per trial, the χ² test also has power to flag
modest discrepancies.)

The second-line trial (133) failed far more severely (χ² *p* = 2×10⁻⁵⁸) and
qualitatively differently: simulated response was roughly fivefold too high (PR
49% versus real 9%) and PD was again under-predicted (3% versus 42%). This is the
anticipated out-of-domain failure. Training was first-line only, with prior
therapy lines held constant, so the model has no representation of second-line
refractory disease and instead applies first-line shrinkage dynamics to a
population that does not exhibit them. It is not a separate failure mode but the
same one—target-lesion dynamics learned in a single setting—pushed past the
boundary of the training distribution.

### 3.4 Overall-survival under-prediction reflects single-trial covariate support

Simulated OS fell below real OS in all four external trials, by 1.1–2.4 months. We
first pursued the most mechanistically natural explanation. The model represents
tumour growth but has no knowledge of treatment after progression, so if the
external trials delivered more post-progression subsequent therapy than the
training population, real OS would be extended in a way the model could not
reproduce. Subsequent-therapy rates did rise across the external trials (from 25%
to 55%), and a naive within-trial comparison appeared to confirm the hypothesis:
patients who received subsequent therapy lived 7–9 months longer than those who
did not.

That comparison, however, is confounded by immortal-time bias—a patient must
survive to progression and remain fit enough to be re-treated, so the untreated
group is enriched for early deaths by construction. A 180-day landmark analysis,
restricting to patients alive at 180 days and classifying by whether subsequent
therapy had begun by then, **reversed** the direction of the association, with
re-treated patients showing *shorter* subsequent survival (12.3 versus 17.7
months in trial 108). Consistent with this reversal, the OS gap across trials
showed no dose-response with subsequent-therapy rate—the correlation was slightly
negative (Spearman ρ = −0.20)—and was, at the level of individual trials, if
anything inverted: the largest gap (trial 141, 2.4 months) occurred in the trial
with the *lowest* subsequent-therapy rate and the same regimen as the training
arm, while the trial with the highest rate (272) showed the *best* OS agreement.
We therefore excluded subsequent therapy as the explanation.

The gap instead tracked each trial's true OS *level*: the Spearman correlation
between the OS gap and the real median OS was **+0.80**. The model under-predicted
precisely those trials whose real OS exceeded the training arm's (13.4, 11.7, and
11.1 months) and matched the one that fell below it (9.9 months). This is the
signature of a survival sub-model that cannot move the OS level between
populations. The training arm offered almost no covariate variation for it to
learn from—one regimen, stage IV only, non-squamous only, and no recorded ECOG
performance status—so its predictions regress toward the training arm's marginal
OS (approximately 10.5 months) regardless of the target population. Two further
observations confirm this reading and locate the cause inside the model rather
than in the external cohorts. First, the under-prediction was already present
internally, where the held-out simulated OS median (approximately 10.8 months) sat
below the real value (approximately 11.8 months); a gap visible within the
training trial itself cannot be an artefact of cross-trial transport. Second, the
single significant external failure (trial 108, *p* = 0.018) reflects statistical
power rather than a population-specific effect—its median gap (1.9 months) was
smaller than that of trial 141 (2.4 months, not significant), and it reached
significance only because of its large sample and high event maturity.

*Figures. Fig. 1: internal held-out PFS/OS Kaplan–Meier overlays and landmark
calibration. Fig. 2: per-trial external OS (and PFS) Kaplan–Meier overlays,
simulated versus real. Fig. 3: per-trial simulated versus real BOR proportions.
All are produced by the validation pipeline (`results/real_data/`).*

---

## 4. Discussion

*[Placeholder — to be drafted after Abstract/Results review. Intended points to
develop:*

- *The contribution is a virtual-control-arm validation in which each negative
  result has an identified mechanism, not merely a reported gap—distinct from
  prior enterprise work that emphasises positive transport.*
- *Two concrete, distinct fixes for two distinct problems: (a) additional
  lesion-level training arms or explicit study-level covariate effects to remove
  the OS compression of §3.4; (b) modelling non-target and new-lesion progression
  events, not target-lesion SLD alone, to restore BOR transport of §3.3.*
- *Single-trial training stated plainly as a current public-data-availability
  constraint (most public sponsors release BOR/survival, not longitudinal SLD),
  with external transport carrying the credibility.*
- *SEER-based population-level real-world calibration is explicitly out of scope
  here and reserved as a separate future addition; the present work is a
  trial-data validation.]*
