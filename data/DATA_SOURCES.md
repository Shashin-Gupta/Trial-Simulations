# Data Sources

This project uses four external data sources. **None of the underlying
patient-level data is redistributed with this repository**, and none of it may
be committed to git (see `.gitignore` and the `LICENSE` data notice). This
document is the operator's manual for obtaining each source yourself.

| Source | Role in this project | Access | Patient-level? |
|---|---|---|---|
| **Project Data Sphere (PDS)** | Primary training data: NSCLC comparator-arm trajectories + outcomes | Manual registration + DUA | Yes |
| **SEER (NCI)** | External calibration: real-world lung-cancer survival | DUA request (~2 business days) | Yes |
| **ClinicalTrials.gov v2 API** | Aggregate benchmark medians (PFS/OS) to sanity-check simulations | Public, no auth | No (summary only) |
| **BioLINCC (NHLBI)** | Not used — see note below | — | — |

---

## 1. Project Data Sphere (PRIMARY, manual download)

**What it is.** A not-for-profit platform (CEO Roundtable on Cancer) hosting
historical, patient-level **Phase II/III comparator-arm** oncology data. NSCLC is
one of the best-represented indications (thousands of lung-cancer patient-lives
across trials studying platinum doublets, docetaxel, erlotinib, bevacizumab,
etc.), which is why it is our first indication. PDS also runs a dedicated
**External Control Arm** program directly aligned with this project's goal.

**Why access is manual.** Data are released under per-dataset use agreements and
account approval is human-gated. There is no programmatic bulk API for the
patient-level files, so **you must download them by hand**; this repo's loader
(`vca.data_processing.project_data_sphere`) reads what you place on disk.

### Registration + download steps

1. Go to <https://www.projectdatasphere.org/> (redirects to
   `data.projectdatasphere.org`) and **create an account** (Research /
   Institutional). Provide your affiliation and intended research use.
2. Wait for account approval (typically a short review).
3. Browse the data catalog and **search for NSCLC / non-small cell lung cancer**
   comparator-arm datasets. Note the trial's NCT id and a short slug you'll use
   as a folder name (e.g. `proclaim_nct00686959`).
4. For each dataset, **accept the dataset-specific Data Use Agreement**. Read it:
   most prohibit re-identification and redistribution.
5. **Download** the provided files. PDS commonly distributes SAS transport
   (`.xpt`), SAS native (`.sas7bdat`), and/or CSV, plus a **data dictionary**.
6. Place them under:
   ```
   data/raw/project_data_sphere/<slug>/
       <files...>.{csv,xpt,sas7bdat}
       data_dictionary.*
   ```

### Turning a raw dataset into the canonical schema

Every sponsor names columns differently, so you supply a small **YAML column
map** (mapping is data, not code). Scaffold a template:

```bash
python -m vca.data_processing.project_data_sphere   # writes data/raw/column_maps/example.yaml
```

Copy it to `data/raw/column_maps/<slug>.yaml`, inspect the raw files (use
`notebooks/00_data_profiling.py`), and edit the right-hand side to the sponsor's
actual column names. Then load:

```python
from vca.data_processing.project_data_sphere import load_project_data_sphere
td = load_project_data_sphere(
    "data/raw/project_data_sphere/<slug>",
    "data/raw/column_maps/<slug>.yaml",
)
td.validate()
td.to_parquet("data/processed", prefix="nsclc")   # stays local, git-ignored
```

**Deriving SLD.** If a dataset provides per-lesion measurements rather than a
precomputed sum of longest diameters, sum target-lesion longest diameters per
patient-visit to produce `sld_mm` before mapping. **Deriving PFS/OS.** If only
dates are provided, compute `*_time_days` from randomization to event/censor and
set the `*_event` indicator (use `event_invert` in the YAML if the source stores
a *censor* flag where 1 = censored).

---

## 2. SEER (external calibration)

**What it is.** NCI's population-based cancer registry — real-world incidence and
survival. We use it **only** as an external sanity check on baseline survival
(is simulated OS in the right neighbourhood for a comparable stage mix?). SEER is
not trial data, has no RECIST trajectories, and is **not** used to fit the model.

**Key nuance:** research microdata comes through the **SEER\*Stat application**
after a data use agreement, **not** through `api.seer.cancer.gov` (that REST
service is for registry/coding integration). As of June 13, 2025, any requestor
with a valid email may access SEER Research Data.

### Steps

1. Request access at <https://seer.cancer.gov/data/access.html>. Approval is
   typically within ~2 business days; you receive a SEER\*Stat account.
2. Install **SEER\*Stat** and sign in.
3. Build a **Case Listing** (or Rate) session for the lung & bronchus cohort you
   want to compare against. Include at least:
   - `Survival months`
   - `Vital status recode (study cutoff used)`
   - `SEER cause-specific death classification` (for cause-specific survival)
   - an AJCC stage variable (to match your trial's stage mix)
4. **Export the matrix/case-listing to CSV** into `data/raw/seer/`.
5. Load and build a comparison curve:
   ```python
   from vca.data_processing.seer import load_seer_caselisting, seer_survival_curve
   df = load_seer_caselisting("data/raw/seer/nsclc_caselisting.csv")
   curve = seer_survival_curve(df, endpoint="overall")   # or "cause_specific"
   ```
   Column labels vary with your SEER\*Stat variable set — pass a `column_map` to
   override the defaults in `vca.data_processing.seer.DEFAULT_SEER_COLUMNS`.

**Validity caveats (call these out in any paper):** SEER is a general population,
not a trial-eligible one (better performance status, fewer comorbidities in
trials → trial arms typically outlive SEER); staging editions and treatment eras
shift over time; SEER lacks RECIST and biomarker detail. Treat SEER as a loose
external anchor, not ground truth.

---

## 3. ClinicalTrials.gov v2 API (aggregate benchmarks — safe to run)

**What it is.** Public registry with **posted aggregate results**. We pull median
PFS/OS per arm for completed NSCLC Phase II/III trials to check that simulated
outcome distributions fall within the historical range. These are summary
statistics, not patient-level data — safe to fetch and (if you wish) share.

No registration, no API key. Run:

```bash
vca-fetch-benchmarks --max-studies 60
# or:  python -m vca.data_processing.clinicaltrials --max-studies 60
```

This writes `data/benchmarks/nsclc_trial_benchmarks.csv` (git-ignored by default
since it is regenerable) and prints the distribution of historical median PFS/OS
across arms. Docs: <https://clinicaltrials.gov/data-api/api>.

---

## 4. BioLINCC (NHLBI) — not used for this indication

BioLINCC hosts NIH-funded **cardiovascular, pulmonary, and hematologic** trial
data. It does **not** hold oncology / NSCLC trial datasets, so it is **out of
scope** for this project. If the indication ever moves to a cardiopulmonary
endpoint it would become relevant; for NSCLC, skip it.
<https://biolincc.nhlbi.nih.gov/>

---

## Data-handling rules (non-negotiable)

- Raw and processed patient-level data live only under `data/raw/` and
  `data/processed/`, both git-ignored. Never commit them.
- Fitted-model artifacts (posterior samples) can encode training data — keep them
  local (`artifacts/`, git-ignored).
- Respect each source's DUA: no re-identification, no redistribution.
