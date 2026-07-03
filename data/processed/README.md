# `data/processed/` — canonical, cleaned datasets (NEVER committed)

Everything here except this README is git-ignored. Loaders in
`src/vca/data_processing/` write canonical-schema Parquet/CSV files here after
mapping raw source exports onto the shared schema
(`vca.data_processing.schema`).

Even though these files are cleaned and de-identified relative to the raw
exports, they are **still patient-level data** governed by the source data use
agreement. They stay on your machine.

Typical contents:

```
data/processed/
├── nsclc_baseline.parquet     # one row per patient (covariates)
├── nsclc_longitudinal.parquet # long format: patient_id, time_days, sld_mm
└── nsclc_events.parquet       # patient_id, pfs_time_days, pfs_event, os_*
```
