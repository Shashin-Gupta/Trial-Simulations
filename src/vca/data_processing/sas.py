"""Robust SAS7BDAT reader used by the real-data (Project Data Sphere) loaders.

Some PDS sponsors export ``.sas7bdat`` files whose compression variant the
``readstat`` C library (via :mod:`pyreadstat`) cannot parse — it raises
``Invalid file, or file has unsupported features``. pandas' pure-Python SAS
reader handles those files, so this helper tries ``pyreadstat`` first (because it
preserves column *labels*, which are invaluable when mapping cryptic sponsor
column names) and falls back to :func:`pandas.read_sas`, recovering labels from
the reader object where possible.

Returned frames are label-decoded: byte strings become ``str`` and are stripped
of the trailing whitespace SAS pads fixed-width character columns with.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _decode(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode("latin-1") if isinstance(v, bytes) else v)
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def read_sas_any(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a ``.sas7bdat`` file, returning ``(dataframe, {column: label})``.

    Works around sponsor files that ``pyreadstat`` cannot open by falling back
    to pandas. Column labels are returned when the reader exposes them (always
    for the ``pyreadstat`` path; best-effort for the pandas fallback).
    """
    path = Path(path)
    try:
        import pyreadstat

        df, meta = pyreadstat.read_sas7bdat(str(path))
        return _decode(df), dict(meta.column_names_to_labels)
    except Exception:
        pass

    labels: dict[str, str] = {}
    try:  # recover labels from the pandas SAS reader's column objects
        reader = pd.read_sas(str(path), format="sas7bdat", iterator=True)
        for col in reader.columns:
            name = col.name.decode() if isinstance(col.name, bytes) else col.name
            label = col.label.decode() if isinstance(col.label, bytes) else col.label
            labels[name] = label
        reader.close()
    except Exception:
        pass

    df = pd.read_sas(str(path), format="sas7bdat", encoding="latin-1")
    return _decode(df), labels
