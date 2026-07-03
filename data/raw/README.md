# `data/raw/` — manually downloaded source data (NEVER committed)

Everything in this directory except this README is git-ignored. **Do not commit
patient-level data here.** Project Data Sphere, SEER, and BioLINCC data use
agreements prohibit redistribution.

Place manually downloaded exports here, one subfolder per source:

```
data/raw/
├── project_data_sphere/
│   └── <dataset_slug>/            # e.g. proclaim_nct00686959/
│       ├── *.csv or *.sas7bdat / *.xpt
│       └── data_dictionary.*
├── seer/
│   └── nsclc_caselisting.csv      # SEER*Stat case-listing export (see DATA_SOURCES.md)
└── column_maps/
    └── <dataset_slug>.yaml        # maps sponsor column names -> canonical schema
```

See `data/DATA_SOURCES.md` for exact registration and download steps, and
`src/vca/data_processing/` for the loaders that read from here.
