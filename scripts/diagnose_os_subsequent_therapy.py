#!/usr/bin/env python
"""Diagnostic: does post-progression subsequent therapy explain the systematic
OS under-prediction seen in the external validation? (methodology.md §10).

Model-free. Measures per-trial subsequent-therapy rates from whatever field each
sponsor provides, splits OS by receipt (with an immortal-time-bias landmark
check), and correlates the sim-vs-real OS gap against therapy rate vs real OS
level. Writes results/real_data/subsequent_therapy_diagnostic.csv.

    python scripts/diagnose_os_subsequent_therapy.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr

from vca.data_processing.pds_trials import load_trial
from vca.data_processing.sas import read_sas_any

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MO = 30.4375


def _os_gap(tid: str) -> tuple[float, float, float]:
    """Canonical (real_os_mo, sim_os_mo, representative logrank_p) from the
    regenerated external validation JSON — not hardcoded, so it tracks the fit."""
    ej = json.loads((ROOT / "results" / "real_data" / "external" / f"{tid}_external.json").read_text())
    o = ej["os"]
    return o["real_median_months"], o["sim_median_months"], o["logrank_p"]


def _med(t, e):
    t = np.asarray(t, float)
    e = np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(e)
    return float(KaplanMeierFitter().fit(t[ok], e[ok].astype(int)).median_survival_time_) / MO


def _os(tid):
    ev = load_trial(tid).data.events.set_index("patient_id")
    return (pd.to_numeric(ev["os_time_days"], errors="coerce"),
            pd.to_numeric(ev["os_event"], errors="coerce"), ev.index)


def _split(tid, flag_ids):
    t, e, idx = _os(tid)
    f = idx.to_series().isin(flag_ids).to_numpy()
    t, e = t.to_numpy(float), e.to_numpy(float)
    lr = logrank_test(t[f], t[~f], event_observed_A=e[f], event_observed_B=e[~f])
    return f.mean(), _med(t[f], e[f]), _med(t[~f], e[~f]), float(lr.p_value)


def rate_141():
    cm = read_sas_any(RAW / "LungNo_EliLill_2008_141/cmtpy.sas7bdat")[0]
    cm["_pid"] = cm["usubjid"].astype(str)
    study = "PACLITAXEL|CARBOPLATIN|BEVACIZUMAB|PEMETREXED|GEMCITABINE"
    is_L = cm["ATC1CD"].astype(str).str.upper().eq("L")
    not_study = ~cm["CMTERM"].astype(str).str.upper().str.contains(study, na=False)
    conmed = cm["trttpsnm"].astype(str).str.upper().isin(["CMTP", "PRVCMTP"])
    return set(cm.loc[is_L & not_study & conmed, "_pid"].unique())


def rate_272():
    adsl = read_sas_any(RAW / "LungNo_EliLill_2010_272/adsl.sas7bdat")[0]
    adsl["_pid"] = adsl["USUBJID"].astype(str)
    return set(adsl.loc[adsl["CMANTCFL"].astype(str).str.upper().eq("Y"), "_pid"])


def rate_133():
    cm = read_sas_any(RAW / "LungNo_SanofiU_2007_133/cm.sas7bdat")[0]
    cm["_pid"] = cm["RUSUBJID"].astype(str)
    anti = cm["CMCAT"].astype(str).str.upper().eq("ANTI-TUMOR THERAPY")
    notd = ~cm["CMDECOD"].astype(str).str.upper().str.contains("DOCETAXEL", na=False)
    day = pd.to_numeric(cm["CMSTDY"], errors="coerce")
    return set(cm.loc[anti & notd & (day > 21), "_pid"].unique())


def rate_108():
    foll = read_sas_any(RAW / "LungNo_Celgene_2007_108/foll.sas7bdat")[0]
    foll["_pid"] = foll["RPT"].astype(str)
    got = foll.groupby("_pid")["THERNEW"].apply(lambda s: s.astype(str).eq("1").any())
    return set(got[got].index)


def landmark_108(day=180.0):
    foll = read_sas_any(RAW / "LungNo_Celgene_2007_108/foll.sas7bdat")[0]
    foll["_pid"] = foll["RPT"].astype(str)
    foll["thd"] = pd.to_numeric(foll["THERDAY"], errors="coerce")
    by = foll[(foll["THERNEW"].astype(str) == "1") & (foll["thd"] <= day)]["_pid"].unique()
    t, e, idx = _os("108")
    t, e = t.to_numpy(float), e.to_numpy(float)
    alive = t >= day
    f = idx.to_series().isin(by).to_numpy()
    sub, non = alive & f, alive & ~f
    lr = logrank_test(t[sub], t[non], event_observed_A=e[sub], event_observed_B=e[non])
    return _med(t[sub], e[sub]), _med(t[non], e[non]), float(lr.p_value)


def main():
    rates = {"141": rate_141(), "272": rate_272(), "133": rate_133(), "108": rate_108()}
    rows = []
    for tid, ids in rates.items():
        rate, med_y, med_n, p = _split(tid, ids)
        real, sim, lp = _os_gap(tid)
        rows.append({"trial": tid, "subseq_rate": round(rate, 3),
                     "os_med_subseq_mo": round(med_y, 1), "os_med_none_mo": round(med_n, 1),
                     "split_logrank_p": p, "real_os_mo": real, "sim_os_mo": sim,
                     "os_gap_mo": round(real - sim, 2), "ext_os_logrank_p": lp})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "real_data" / "subsequent_therapy_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    r_rate, p_rate = spearmanr(df["subseq_rate"], df["os_gap_mo"])
    r_os, p_os = spearmanr(df["real_os_mo"], df["os_gap_mo"])
    print(f"\nSpearman  OS gap ~ subsequent-therapy rate : rho={r_rate:+.2f} (p={p_rate:.2f})")
    print(f"Spearman  OS gap ~ real OS median          : rho={r_os:+.2f} (p={p_os:.2f})")
    ly, ln, lp = landmark_108()
    print(f"\n108 immortal-time-bias landmark (alive at 180d): subseq-by-180 {ly:.1f}mo "
          f"vs none {ln:.1f}mo (p={lp:.1e}) — naive split was 15.8 vs 6.9mo")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
