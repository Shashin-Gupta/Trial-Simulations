"""Loaders for the five real Project Data Sphere (PDS) NSCLC trials.

These map four *different* sponsor export formats onto the project's canonical
schema (:class:`~vca.data_processing.schema.TrialData`) plus an aggregate
best-overall-response (BOR) series. PDS shares the **control/comparator arm
only**, so each file set is effectively a single treatment arm.

The revised validation design (see ``docs/methodology.md``):

* ``_438`` (Eli Lilly H3E-US-S130) is the **only** trial with patient-level
  RECIST lesion trajectories, so the model is trained and internally validated
  on it alone. :func:`load_438` returns a full ``TrialData`` (baseline +
  longitudinal SLD + events).
* The other four trials (``141``, ``272``, ``133``, ``108``) provide baseline
  covariates + aggregate BOR + PFS/OS only; they are used for **external
  aggregate validation**. Their loaders return a :class:`RealTrial` with an
  empty longitudinal table.

Raw files live under ``data/raw/<trial_id>/`` and are never committed (DUA).
Every sponsor names and codes its columns differently, so the per-trial mapping
is explicit code here rather than a generic YAML: the messy legacy formats (e.g.
CA031) need real logic (date arithmetic, multi-file survival assembly,
frequency-checked code maps), not just column renames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vca.data_processing.sas import read_sas_any
from vca.data_processing.schema import (
    BASELINE_REQUIRED,
    EVENTS_REQUIRED,
    LONGITUDINAL_REQUIRED,
    TrialData,
    coerce_dtypes,
)

DAYS_PER_MONTH = 30.4375
DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"

TRIAL_DIRS = {
    "438": "LungNo_EliLill_2009_438",
    "141": "LungNo_EliLill_2008_141",
    "272": "LungNo_EliLill_2010_272",
    "133": "LungNo_SanofiU_2007_133",
    "108": "LungNo_Celgene_2007_108",
}

# BOR categories the simulator can produce; "NE" = not evaluable (real only).
BOR_LEVELS = ("CR", "PR", "SD", "PD")


@dataclass
class RealTrial:
    """A single real trial mapped to canonical form plus aggregate BOR.

    Attributes
    ----------
    trial_id
        Short id, e.g. ``"272"``.
    data
        Canonical :class:`TrialData`. ``longitudinal`` is empty for the four
        external-validation trials (no lesion-level data available/used).
    bor
        One row per patient: best overall response in ``{CR, PR, SD, PD, NE}``.
    meta
        Descriptive fields (regimen, therapy line, histology label, censoring
        convention, whether PFS is reliably derivable, etc.).
    """

    trial_id: str
    data: TrialData
    bor: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.data.n_patients

    def bor_proportions(self, include_ne: bool = False) -> pd.Series:
        """BOR proportions. By default computed over evaluable patients only."""
        levels = list(BOR_LEVELS) + (["NE"] if include_ne else [])
        s = self.bor if include_ne else self.bor[self.bor.isin(BOR_LEVELS)]
        counts = s.value_counts().reindex(levels, fill_value=0)
        return counts / counts.sum() if counts.sum() else counts.astype(float)


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #

def _dir(trial_id: str) -> Path:
    return DATA_ROOT / TRIAL_DIRS[trial_id]


def _read(trial_id: str, stem: str) -> pd.DataFrame:
    return read_sas_any(_dir(trial_id) / f"{stem}.sas7bdat")[0]


def _as_ts(series: pd.Series) -> pd.Series:
    """Coerce a date/datetime column (date objects or Timestamps) to Timestamp."""
    return pd.to_datetime(series, errors="coerce")


def _u(series: pd.Series) -> pd.Series:
    """Uppercase-stripped string view of a column."""
    return series.astype("string").str.strip().str.upper()


def _empty_longitudinal() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LONGITUDINAL_REQUIRED})


def _canonical_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every canonical baseline column exists (missing -> NA)."""
    out = df.copy()
    for c in BASELINE_REQUIRED:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def _norm_bor_label(x) -> str:
    """Map assorted response labels to {CR, PR, SD, PD, NE}."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NE"
    s = str(x).strip().upper()
    if s in ("CR", "COMPLETE RESPONSE"):
        return "CR"
    if s in ("PR", "PARTIAL RESPONSE"):
        return "PR"
    if s in ("SD", "STABLE DISEASE"):
        return "SD"
    if s in ("PD", "PROGRESSIVE DISEASE", "PROGRESSION"):
        return "PD"
    if s in ("IR", "IR/SD", "SD/IR", "INCOMPLETE RESPONSE/STABLE DISEASE"):
        return "SD"  # RECIST 1.0 "Incomplete Response/Stable Disease" bucket
    return "NE"  # UNK, ND, NE, "Not Evaluable", blank, ...


def _bor_from_ordinal(codes: pd.Series) -> str:
    """Best (= most favourable) RECIST response from ordinal codes 1..4."""
    vals = pd.to_numeric(codes, errors="coerce").dropna()
    vals = vals[vals.isin([1, 2, 3, 4])]
    if vals.empty:
        return "NE"
    return {1: "CR", 2: "PR", 3: "SD", 4: "PD"}[int(vals.min())]


# --------------------------------------------------------------------------- #
# _438 — Eli Lilly H3E-US-S130 (training trial, full longitudinal)
# --------------------------------------------------------------------------- #

def _histology_lilly(snm) -> str:
    s = "" if snm is None else str(snm).strip().lower()
    if "squamous" in s:
        return "squamous"
    if "adeno" in s or "large cell" in s:
        return "non_squamous"
    return "other"


def _smoking_lilly(smk_ever, tobacco_now) -> str:
    s = "" if smk_ever is None else str(smk_ever).strip().lower()
    if s.startswith("never"):
        return "never"
    if s.startswith("ever"):
        return "current" if pd.notna(tobacco_now) and float(tobacco_now) == 1 else "former"
    return "unknown"


def load_438() -> RealTrial:
    """Full canonical bundle for _438, the training trial (with SLD trajectories)."""
    subj = _read("438", "subjinfo")
    diag = _read("438", "diag")
    tte = _read("438", "ttevent")
    bor_raw = _read("438", "bor")
    les = _read("438", "lesions")

    pid = subj["USUBJID"].astype(str)
    rand = _as_ts(subj["RNDMDT"])
    stage_map = (
        diag.assign(_pid=diag["USUBJID"].astype(str))
        .set_index("_pid")["DISSTGSNM"].astype(str).str.replace("Stage ", "", regex=False)
    )
    baseline = pd.DataFrame({
        "patient_id": pid.values,
        "age": pd.to_numeric(subj["AGEYR"], errors="coerce").values,
        "sex": _u(subj["SEXSNM"]).values,
        "ecog_ps": pd.NA,  # not recorded in this trial
        "stage": stage_map.reindex(pid.values).values,
        "histology": [_histology_lilly(h) for h in subj["HSTYPESNM"]],
        "smoking": [_smoking_lilly(a, b)
                    for a, b in zip(subj["SMKSTATSNM"], subj["TBBL"], strict=False)],
        "prior_lines": 0,  # first-line trial
        "study_id": "PDS-438-H3E-US-S130",
        "treatment": "Paclitaxel+Carboplatin+Bevacizumab",
    })

    # --- longitudinal SLD from target lesions (LSRN in cm -> mm) -------------
    tgt = les[_u(les["LSTPSNM"]) == "TARGET"].copy()
    tgt["_pid"] = tgt["USUBJID"].astype(str)
    tgt["_sld_mm"] = pd.to_numeric(tgt["LSRN"], errors="coerce") * 10.0
    tgt = tgt.dropna(subset=["_sld_mm"])
    tgt["_date"] = _as_ts(tgt["LSASMDT"])
    per_visit = (
        tgt.dropna(subset=["_date"])
        .groupby(["_pid", "_date"], as_index=False)
        .agg(sld_mm=("_sld_mm", "sum"), n_lesions=("_sld_mm", "size"))
    )
    per_visit["time_days"] = (
        per_visit["_date"].values - rand.set_axis(pid.values).reindex(per_visit["_pid"]).values
    ) / np.timedelta64(1, "D")
    # Keep assessments from the screening window onward; drop rare mis-dated
    # historical scans (a few sit hundreds of days pre-randomization).
    per_visit = per_visit[(per_visit["time_days"] >= -60) & (per_visit["time_days"] <= 1095)]
    longitudinal = per_visit.rename(columns={"_pid": "patient_id"})[
        ["patient_id", "time_days", "sld_mm"]
    ]
    longitudinal = longitudinal[longitudinal["time_days"].notna()]

    # baseline SLD anchor + target-lesion count: the assessment nearest t=0
    per_visit = per_visit.assign(_abst=per_visit["time_days"].abs())
    first = per_visit.sort_values("_abst").groupby("_pid", as_index=False).first()
    baseline = baseline.merge(
        first[["_pid", "sld_mm", "n_lesions"]].rename(
            columns={"_pid": "patient_id", "sld_mm": "baseline_sld_mm",
                     "n_lesions": "n_target_lesions"}),
        on="patient_id", how="left",
    )

    events = _lilly_events(tte, pid.values, months=True)
    bor = _lilly_bor(bor_raw)

    data = TrialData(baseline=_canonical_baseline(baseline),
                     longitudinal=longitudinal, events=events)
    coerce_dtypes(data)
    data.validate(strict=False)
    meta = dict(
        regimen="Paclitaxel+Carboplatin+Bevacizumab", line="1L",
        histology_label="non-squamous", n=data.n_patients, pfs_available=True,
        pfs_convention="RECIST PFS, censored at subsequent anti-cancer therapy",
        os_convention="death from any cause, administrative censoring",
        reader="pyreadstat",
    )
    return RealTrial("438", data, bor, meta)


def _lilly_events(tte: pd.DataFrame, order, *, months: bool) -> pd.DataFrame:
    """Assemble PFS/OS from an Eli Lilly ``ttevent`` long table (438/141)."""
    tte = tte.copy()
    tte["_pid"] = tte["USUBJID"].astype(str)
    code = _u(tte["TTECD"])
    scale = DAYS_PER_MONTH if months else 1.0

    def _one(endpoint_code):
        sub = tte[code == endpoint_code]
        t = pd.to_numeric(sub["TTERN"], errors="coerce") * scale
        cens = pd.to_numeric(sub["TTECENSFLG"], errors="coerce")
        return (pd.DataFrame({"patient_id": sub["_pid"].values,
                              "time": t.values, "event": (1 - cens).values})
                .dropna(subset=["patient_id"]).drop_duplicates("patient_id")
                .set_index("patient_id"))

    pfs, os_ = _one("PFS"), _one("OS")
    ev = pd.DataFrame({"patient_id": [str(p) for p in order]}).set_index("patient_id")
    ev["pfs_time_days"] = pfs["time"]
    ev["pfs_event"] = pfs["event"]
    ev["os_time_days"] = os_["time"]
    ev["os_event"] = os_["event"]
    return ev.reset_index()[EVENTS_REQUIRED]


def _lilly_bor(bor_raw: pd.DataFrame) -> pd.Series:
    b = bor_raw.copy()
    b["_pid"] = b["USUBJID"].astype(str)
    s = b.drop_duplicates("_pid").set_index("_pid")["BORCDSNM"].map(_norm_bor_label)
    s.name = "bor"
    return s


# --------------------------------------------------------------------------- #
# 141 — Eli Lilly JMHD (same schema as _438; times in DAYS; no lesions)
# --------------------------------------------------------------------------- #

def load_141() -> RealTrial:
    subj = _read("141", "subjinfo")
    tte = _read("141", "ttevent")
    bor_raw = _read("141", "bor")

    pid = subj["usubjid"].astype(str)
    baseline = pd.DataFrame({
        "patient_id": pid.values,
        "age": pd.to_numeric(subj["ageyr"], errors="coerce").values,
        "sex": _u(subj["sexsnm"]).values,
        "ecog_ps": pd.NA,
        "stage": "IV",  # advanced/metastatic first-line cohort (no explicit stage col)
        "histology": [_histology_lilly(h) for h in subj["hstypesnm"]],
        "smoking": [_smoking_lilly(a, None) for a in subj["smkstat"]],
        "prior_lines": 0,
        "study_id": "PDS-141-H3E-MC-JMHD",
        "treatment": "Paclitaxel+Carboplatin+Bevacizumab",
    })
    # ttevent here uses upper-case TTECD/TTERN but lower-case usubjid
    tte = tte.rename(columns={"usubjid": "USUBJID", "ttecensflg": "TTECENSFLG"})
    events = _lilly_events(tte, pid.values, months=False)  # TTERULNM == 'day'
    bor = _lilly_bor(bor_raw.rename(columns={"usubjid": "USUBJID"}))

    data = TrialData(baseline=_canonical_baseline(baseline),
                     longitudinal=_empty_longitudinal(), events=events)
    coerce_dtypes(data)
    data.validate(strict=False)
    meta = dict(
        regimen="Paclitaxel+Carboplatin+Bevacizumab", line="1L",
        histology_label="non-squamous", n=data.n_patients, pfs_available=True,
        pfs_convention="RECIST PFS, censored at subsequent anti-cancer therapy",
        os_convention="death from any cause, administrative censoring",
        reader="pandas-fallback",
    )
    return RealTrial("141", data, bor, meta)


# --------------------------------------------------------------------------- #
# 272 — Eli Lilly JFCC / SQUIRE (ADaM; squamous; Gemcitabine+Cisplatin)
# --------------------------------------------------------------------------- #

def _smoking_squire(x) -> str:
    s = "" if x is None else str(x).strip().lower()
    if "non smoker" in s or "never" in s or "fewer than 100" in s:
        return "never"
    if "smoker" in s:
        return "former"  # SMKB does not separate current vs former cleanly
    return "unknown"


def load_272() -> RealTrial:
    adsl = _read("272", "adsl")
    adtte = _read("272", "adtte")
    adresp = _read("272", "adresp")

    pid = adsl["USUBJID"].astype(str)
    stage_iv = _u(adsl["DISSTGFL"]) == "N"  # "No Stage IV Disease" == N -> is stage IV
    baseline = pd.DataFrame({
        "patient_id": pid.values,
        "age": pd.to_numeric(adsl["AGE"], errors="coerce").values,
        "sex": _u(adsl["SEX"]).values,
        "ecog_ps": pd.to_numeric(adsl["ECOGBL"], errors="coerce").values,
        "stage": np.where(stage_iv, "IV", "III"),
        "histology": "squamous",  # SQUIRE enrolled squamous NSCLC only
        "smoking": [_smoking_squire(x) for x in adsl["SMKB"]],
        "prior_lines": 0,
        "study_id": "PDS-272-CP11-0806-SQUIRE",
        "treatment": "Gemcitabine+Cisplatin",
    })
    events = _adam_events(adtte, pid.values)
    best = adresp[_u(adresp["PARAMCD"]) == "BESTRESP"].copy()
    best["_pid"] = best["USUBJID"].astype(str)
    bor = best.drop_duplicates("_pid").set_index("_pid")["AVALC"].map(_norm_bor_label)
    bor.name = "bor"

    data = TrialData(baseline=_canonical_baseline(baseline),
                     longitudinal=_empty_longitudinal(), events=events)
    coerce_dtypes(data)
    data.validate(strict=False)
    meta = dict(
        regimen="Gemcitabine+Cisplatin", line="1L", histology_label="squamous",
        n=data.n_patients, pfs_available=True,
        pfs_convention="RECIST PFS (ADaM adtte PARAMCD=PFS), CNSR=1 censored",
        os_convention="OS (ADaM adtte PARAMCD=OS), CNSR=1 censored",
        reader="pyreadstat",
    )
    return RealTrial("272", data, bor, meta)


def _adam_events(adtte: pd.DataFrame, order) -> pd.DataFrame:
    """PFS/OS from an ADaM ``adtte`` long table (AVAL months, CNSR 1=censored)."""
    a = adtte.copy()
    a["_pid"] = a["USUBJID"].astype(str)
    pc = _u(a["PARAMCD"])

    def _one(code):
        sub = a[pc == code]
        t = pd.to_numeric(sub["AVAL"], errors="coerce") * DAYS_PER_MONTH
        cens = pd.to_numeric(sub["CNSR"], errors="coerce")
        return (pd.DataFrame({"patient_id": sub["_pid"].values,
                              "time": t.values, "event": (1 - cens).values})
                .drop_duplicates("patient_id").set_index("patient_id"))

    pfs, os_ = _one("PFS"), _one("OS")
    ev = pd.DataFrame({"patient_id": [str(p) for p in order]}).set_index("patient_id")
    ev["pfs_time_days"] = pfs["time"]
    ev["pfs_event"] = pfs["event"]
    ev["os_time_days"] = os_["time"]
    ev["os_event"] = os_["event"]
    return ev.reset_index()[EVENTS_REQUIRED]


# --------------------------------------------------------------------------- #
# 133 — Sanofi VITAL EFC10261 (SDTM; 2nd-line; Placebo+Docetaxel)
# --------------------------------------------------------------------------- #

def _histology_sdtm(x) -> str:
    s = "" if x is None else str(x).strip().lower()
    if "squamous" in s or "epidermoid" in s:
        return "squamous"
    if "adeno" in s or "large cell" in s or "bronchoalveolar" in s or "non-small" in s:
        return "non_squamous"
    return "other"


def load_133() -> RealTrial:
    dm = _read("133", "dm")
    ds = _read("133", "ds")
    cd = _read("133", "cd")
    su = _read("133", "su")
    ls = _read("133", "ls")

    pid = dm["RUSUBJID"].astype(str)
    # histology + stage from the cancer-diagnosis (CD) domain
    cd = cd.copy()
    cd["_pid"] = cd["RUSUBJID"].astype(str)
    tc = _u(cd["CDTESTCD"])
    hist = cd[tc == "HTYPE"].drop_duplicates("_pid").set_index("_pid")["CDSTRESC"]
    stage = (cd[tc == "STAGE"].drop_duplicates("_pid").set_index("_pid")["CDSTRESC"]
             .astype("string").str.replace("STAGE ", "", case=False, regex=False)
             .str.replace(" ", "", regex=False).str.strip())  # "III B" -> "IIIB"
    # smoking from substance-use (SU) domain: SUSMKST appears on any SU row
    su = su.copy()
    su["_pid"] = su["RUSUBJID"].astype(str)
    smk = (su.dropna(subset=["SUSMKST"]).drop_duplicates("_pid")
           .set_index("_pid")["SUSMKST"].map(
               lambda x: {"NEVER": "never", "CURRENT": "current", "FORMER": "former"}
               .get(str(x).strip().upper(), "unknown")))

    baseline = pd.DataFrame({
        "patient_id": pid.values,
        "age": pd.to_numeric(dm["AGE"], errors="coerce").values,
        "sex": _u(dm["SEX"]).values,
        "ecog_ps": pd.NA,  # not present in provided SDTM domains
        "stage": stage.reindex(pid.values).values,
        "histology": [_histology_sdtm(h) for h in hist.reindex(pid.values).values],
        "smoking": smk.reindex(pid.values).values,
        "prior_lines": 1,  # second-line: all patients had prior platinum therapy
        "study_id": "PDS-133-EFC10261-VITAL",
        "treatment": "Placebo+Docetaxel",
    })

    events = _sdtm_events_133(dm, ds, pid.values)
    # BOR from the lesion/response (LS) domain, LSTESTCD == BORESP
    ls = ls.copy()
    ls["_pid"] = ls["RUSUBJID"].astype(str)
    boresp = ls[_u(ls["LSTESTCD"]) == "BORESP"]
    bor = boresp.drop_duplicates("_pid").set_index("_pid")["LSSTRESC"].map(_norm_bor_label)
    bor = bor.reindex(pid.values).fillna("NE")
    bor.index = pid.values
    bor.name = "bor"

    data = TrialData(baseline=_canonical_baseline(baseline),
                     longitudinal=_empty_longitudinal(), events=events)
    coerce_dtypes(data)
    data.validate(strict=False)
    meta = dict(
        regimen="Placebo+Docetaxel", line="2L", histology_label="non-squamous (mixed)",
        n=data.n_patients, pfs_available=False,
        pfs_convention="not derived — SDTM export lacks a populated progression date (DSSTDY empty)",
        os_convention="OS time = DM.RFENDY (reference end day), event = DS DEATH disposition",
        reader="pandas-fallback",
    )
    return RealTrial("133", data, bor, meta)


def _sdtm_events_133(dm: pd.DataFrame, ds: pd.DataFrame, order) -> pd.DataFrame:
    """Derive OS for the Sanofi VITAL SDTM export (PFS intentionally omitted).

    ``DSSTDY`` is not populated, so OS uses ``DM.RFENDY`` (subject reference end
    day = last day of study participation, i.e. death day for those who died) as
    the time, with the event flag set from a DEATH disposition in ``DS``. There
    is no populated progression date, so a derived PFS would collapse onto OS and
    is deliberately left missing (see methodology.md).
    """
    idx = [str(p) for p in order]
    dm = dm.copy()
    dm["_pid"] = dm["RUSUBJID"].astype(str)
    rfend = pd.to_numeric(dm.set_index("_pid")["RFENDY"], errors="coerce").reindex(idx)
    os_time = rfend.clip(lower=1.0)

    d = ds.copy()
    d["_pid"] = d["RUSUBJID"].astype(str)
    decod = _u(d["DSDECOD"])
    dead_ids = set(d.loc[decod.isin(["DEAD", "DEATH", "DIED"]), "_pid"])
    os_event = pd.Series([1 if p in dead_ids else 0 for p in idx], index=idx)

    ev = pd.DataFrame({
        "patient_id": idx,
        "pfs_time_days": np.nan,
        "pfs_event": pd.array([pd.NA] * len(idx), dtype="Int64"),
        "os_time_days": os_time.values,
        "os_event": os_event.values,
    })
    return ev[EVENTS_REQUIRED]


# --------------------------------------------------------------------------- #
# 108 — Celgene/Abraxis CA031 (legacy; 1st-line; solvent Paclitaxel+Carboplatin)
# --------------------------------------------------------------------------- #

_CA031_HIST = {"1": "non_squamous", "2": "squamous", "3": "non_squamous", "4": "other"}
_CA031_STAGE = {"2": "IIIB", "3": "IV"}


def _smoking_ca031(hx, now) -> str:
    hx = "" if hx is None else str(hx).strip()
    now = "" if now is None else str(now).strip()
    if hx == "1":
        return "never"
    if hx == "2":
        return {"1": "current", "2": "former"}.get(now, "former")
    return "unknown"


def load_108() -> RealTrial:
    demo = _read("108", "demo")
    cnhx = _read("108", "cnhx")
    smok = _read("108", "smok")
    ecog = _read("108", "ecog")
    eos = _read("108", "eos")
    foll = _read("108", "foll")
    resp = _read("108", "resp")

    pid = demo["RPT"].astype(str)
    age = -pd.to_numeric(demo["DOBDAY"], errors="coerce") / 365.25
    sex = _u(demo["GENDER"]).map({"1": "M", "2": "F"})

    cnhx = cnhx.copy()
    cnhx["_pid"] = cnhx["RPT"].astype(str)
    hist = cnhx.drop_duplicates("_pid").set_index("_pid")["DIAGHIST"].astype("string").str.strip()
    stage = cnhx.drop_duplicates("_pid").set_index("_pid")["CSTAGE"].astype("string").str.strip()
    smok = smok.copy()
    smok["_pid"] = smok["RPT"].astype(str)
    smk = smok.drop_duplicates("_pid").set_index("_pid").apply(
        lambda r: _smoking_ca031(r.get("SMOKEHX"), r.get("SMOKE")), axis=1)

    # baseline ECOG = earliest assessment (smallest ASSESSDAY) per patient
    ec = ecog.copy()
    ec["_pid"] = ec["RPT"].astype(str)
    ec["_day"] = pd.to_numeric(ec["ASSESSDAY"], errors="coerce")
    ec["_score"] = pd.to_numeric(ec["ECOGSCOR"], errors="coerce")
    ecog_bl = (ec.dropna(subset=["_score"]).sort_values("_day")
               .groupby("_pid").first()["_score"])

    baseline = pd.DataFrame({
        "patient_id": pid.values,
        "age": age.values,
        "sex": sex.values,
        "ecog_ps": ecog_bl.reindex(pid.values).values,
        "stage": stage.reindex(pid.values).map(_CA031_STAGE).values,
        "histology": hist.reindex(pid.values).map(_CA031_HIST).values,
        "smoking": smk.reindex(pid.values).values,
        "prior_lines": 0,
        "study_id": "PDS-108-CA031",
        "treatment": "Paclitaxel+Carboplatin",
    })

    events = _ca031_events(eos, foll, pid.values)
    bor = _ca031_bor(resp, pid.values)

    data = TrialData(baseline=_canonical_baseline(baseline),
                     longitudinal=_empty_longitudinal(), events=events)
    coerce_dtypes(data)
    data.validate(strict=False)
    meta = dict(
        regimen="Paclitaxel+Carboplatin (solvent-based)", line="1L",
        histology_label="mixed (~42% squamous)", n=data.n_patients, pfs_available=True,
        pfs_convention="derived: date of progression or death (EOS/FOLL) vs censoring at last contact",
        os_convention="derived: death (ALIVEYN/DEATHDAY) vs censoring at last contact",
        reader="pyreadstat",
    )
    return RealTrial("108", data, bor, meta)


def _ca031_events(eos: pd.DataFrame, foll: pd.DataFrame, order) -> pd.DataFrame:
    """OS/PFS from CA031 end-of-study + follow-up forms (days from eligibility).

    Times are 'days between <event> and eligibility verification' fields. Most
    deaths are recorded on the repeated follow-up (FOLL) rows, not the single
    end-of-study (EOS) row, so death/contact/progression days are pooled across
    both. A patient is counted dead if any row carries a positive death day or an
    ALIVEYN='2' flag; the death day is the latest such report, and last contact
    is the latest successful-contact / last-contact day.
    """
    def pool(df):
        d = df.copy()
        d["_pid"] = d["RPT"].astype(str)
        cols = {"_pid": d["_pid"]}
        cols["alive"] = _u(d["ALIVEYN"]) if "ALIVEYN" in d else pd.Series(pd.NA, index=d.index)
        for src, dst in [("DEATHDAY", "death"), ("LCONDAY", "lcon"),
                         ("CONTDAY", "cont"), ("PROGDAY", "prog")]:
            cols[dst] = pd.to_numeric(d[src], errors="coerce") if src in d else np.nan
        return pd.DataFrame(cols)

    pooled = pd.concat([pool(eos), pool(foll)], ignore_index=True)
    idx = [str(p) for p in order]
    g = pooled.groupby("_pid")

    death_pos = pooled.assign(death=pooled["death"].where(pooled["death"] > 0))
    death_day = death_pos.groupby("_pid")["death"].max().reindex(idx)
    dead = (g["alive"].apply(lambda s: (s == "2").any()).reindex(idx).fillna(False)
            | death_day.notna())
    last_contact = pd.concat(
        [g["cont"].max(), g["lcon"].max()], axis=1
    ).max(axis=1).reindex(idx)
    prog_day = (pooled.assign(prog=pooled["prog"].where(pooled["prog"] > 0))
                .groupby("_pid")["prog"].min().reindex(idx))

    os_time = death_day.where(dead, last_contact)
    os_event = dead.astype(int)

    prog_or_death = pd.concat([prog_day, death_day.where(dead)], axis=1).min(axis=1)
    pfs_event = prog_or_death.notna().astype(int)
    pfs_time = prog_or_death.where(prog_or_death.notna(), last_contact)

    ev = pd.DataFrame({
        "patient_id": idx,
        "pfs_time_days": pd.to_numeric(pfs_time, errors="coerce").clip(lower=1.0).values,
        "pfs_event": pfs_event.values,
        "os_time_days": pd.to_numeric(os_time, errors="coerce").clip(lower=1.0).values,
        "os_event": os_event.values,
    })
    return ev[EVENTS_REQUIRED]


def _ca031_bor(resp: pd.DataFrame, order) -> pd.Series:
    r = resp.copy()
    r["_pid"] = r["RPT"].astype(str)
    best = r.groupby("_pid")["OVERRESP"].apply(_bor_from_ordinal)
    idx = [str(p) for p in order]
    bor = best.reindex(idx).fillna("NE")
    bor.index = idx
    bor.name = "bor"
    return bor


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

_LOADERS = {"438": load_438, "141": load_141, "272": load_272,
            "133": load_133, "108": load_108}
VALIDATION_TRIALS = ("141", "272", "133", "108")


def load_trial(trial_id: str) -> RealTrial:
    if trial_id not in _LOADERS:
        raise KeyError(f"unknown trial_id {trial_id!r}; expected one of {list(_LOADERS)}")
    return _LOADERS[trial_id]()


def load_all() -> dict[str, RealTrial]:
    return {tid: load_trial(tid) for tid in _LOADERS}
