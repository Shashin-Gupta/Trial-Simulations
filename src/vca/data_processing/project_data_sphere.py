"""Loader for manually downloaded Project Data Sphere (PDS) comparator-arm data.

Project Data Sphere access is human-gated: you register, accept a data use
agreement, and download the dataset yourself (see ``data/DATA_SOURCES.md``).
This module therefore does **not** download anything. It assumes you have placed
the raw exports under ``data/raw/project_data_sphere/<slug>/`` and mapped the
sponsor's column names onto the canonical schema via a small YAML file.

Because every sponsor names its columns differently, the mapping is data, not
code: write one ``column_maps/<slug>.yaml`` per dataset. A template can be
scaffolded with :func:`write_example_column_map`.

    from vca.data_processing.project_data_sphere import load_project_data_sphere
    td = load_project_data_sphere("data/raw/project_data_sphere/proclaim",
                                  "data/raw/column_maps/proclaim.yaml")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from vca.data_processing.schema import TrialData, coerce_dtypes

CANONICAL_TABLES = ("baseline", "longitudinal", "events")


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV / SAS transport (.xpt) / SAS native (.sas7bdat) into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix == ".xpt":
        # SAS transport format is supported by pandas directly.
        return pd.read_sas(path, format="xport")
    if suffix == ".sas7bdat":
        try:
            import pyreadstat  # optional [sas] extra; preserves labels/formats
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading .sas7bdat needs the [sas] extra: pip install -e '.[sas]'"
            ) from exc
        df, _meta = pyreadstat.read_sas7bdat(str(path))
        return df
    if suffix in (".parquet",):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def _apply_table_spec(dataset_dir: Path, spec: dict) -> pd.DataFrame:
    """Build one canonical table from its mapping spec."""
    raw = _read_any(dataset_dir / spec["file"])
    columns: dict[str, str] = spec["columns"]  # canonical -> source
    missing = [src for src in columns.values() if src not in raw.columns]
    if missing:
        raise KeyError(
            f"{spec['file']}: source columns not found: {missing}. "
            f"Available: {list(raw.columns)[:20]}..."
        )
    # Select and rename to canonical names.
    out = raw[list(columns.values())].copy()
    out.columns = list(columns.keys())

    # Optional per-column value recoding (e.g. numeric sex codes -> M/F).
    for col, mapping in (spec.get("value_maps") or {}).items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: mapping.get(v, v))

    # Optional event-flag inversion: many sponsors store a *censor* flag where
    # 1 = censored; the canonical schema wants 1 = event.
    for col in spec.get("event_invert") or []:
        if col in out.columns:
            out[col] = 1 - pd.to_numeric(out[col], errors="coerce")

    return out


def load_project_data_sphere(dataset_dir: str | Path, column_map_path: str | Path) -> TrialData:
    """Assemble a canonical :class:`TrialData` from a PDS dataset + mapping YAML.

    The YAML must define ``tables.baseline``, ``tables.longitudinal`` and
    ``tables.events``, each with a ``file`` and a ``columns`` map from canonical
    names to the sponsor's source column names.
    """
    dataset_dir = Path(dataset_dir)
    with open(column_map_path) as fh:
        cfg = yaml.safe_load(fh)
    tables = cfg.get("tables", {})
    for t in CANONICAL_TABLES:
        if t not in tables:
            raise KeyError(f"column map missing 'tables.{t}' section")

    baseline = _apply_table_spec(dataset_dir, tables["baseline"])
    longitudinal = _apply_table_spec(dataset_dir, tables["longitudinal"])
    events = _apply_table_spec(dataset_dir, tables["events"])

    # Stamp the study id if provided at top level.
    if "study_id" in cfg and "study_id" not in baseline.columns:
        baseline["study_id"] = cfg["study_id"]

    td = TrialData(baseline=baseline, longitudinal=longitudinal, events=events)
    coerce_dtypes(td)
    return td.validate(strict=False)


EXAMPLE_COLUMN_MAP = """\
# Project Data Sphere column map. Copy to data/raw/column_maps/<slug>.yaml and
# edit the right-hand side to match this dataset's actual column names. Inspect
# the raw files first (e.g. `notebooks/00_data_profiling`) to discover them.
study_id: PDS-<NCTID>
tables:
  baseline:
    file: demographics.csv          # relative to the dataset directory
    columns:                        # canonical_name: SOURCE_COLUMN
      patient_id: SUBJID
      age: AGE
      sex: SEX
      ecog_ps: ECOGBL
      stage: STAGE
      histology: HISTOLOGY
      smoking: SMKSTAT
      prior_lines: NPRIOR
      baseline_sld_mm: SLDBL
      n_target_lesions: NTLESION
    value_maps:                     # optional recoding of source values
      sex: {1: M, 2: F}
      histology:
        "Squamous cell carcinoma": squamous
        "Adenocarcinoma": non_squamous
  longitudinal:
    file: tumor_measurements.csv
    columns:
      patient_id: SUBJID
      time_days: TMDY               # study day of the RECIST assessment
      sld_mm: SUMTL                 # sum of target-lesion longest diameters
  events:
    file: survival.csv
    columns:
      patient_id: SUBJID
      pfs_time_days: PFSDY
      pfs_event: PFSCNSR            # if this is a *censor* flag, list it below
      os_time_days: OSDY
      os_event: DTH
    event_invert: [pfs_event]       # remove if PFS flag is already 1 = event
"""


def write_example_column_map(path: str | Path = "data/raw/column_maps/example.yaml") -> Path:
    """Write a template column-map YAML the user can copy and edit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_COLUMN_MAP)
    return path


if __name__ == "__main__":  # pragma: no cover
    p = write_example_column_map()
    print(f"Wrote template column map -> {p}")
