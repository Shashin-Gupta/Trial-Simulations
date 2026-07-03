"""ClinicalTrials.gov v2 API client for aggregate trial-outcome benchmarks.

This pulls *published, aggregate* results (median PFS/OS per arm) for the target
indication from the public ClinicalTrials.gov API. These are not patient-level
data — they are summary statistics from completed trials — and they are used to
sanity-check that the simulator's outcome distributions land in a clinically
realistic range (e.g. simulated median PFS should sit within the spread of
historical comparator arms).

The API needs no key and no auth. This module is safe to run out of the box:

    python -m vca.data_processing.clinicaltrials --max-studies 60
    # or, once installed:  vca-fetch-benchmarks --max-studies 60

Docs: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://clinicaltrials.gov/api/v2/studies"
MONTHS_TO_DAYS = 30.4375
WEEKS_TO_DAYS = 7.0

_PFS = re.compile(r"progression[\s-]*free survival|(?<![a-z])pfs(?![a-z])", re.I)
_OS = re.compile(r"overall survival|(?<![a-z])os(?![a-z])", re.I)


def fetch_studies(
    condition: str = "non-small cell lung cancer",
    *,
    phases: tuple[str, ...] = ("2", "3"),
    with_results: bool = True,
    max_studies: int = 60,
    page_size: int = 50,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch full study records for ``condition`` from ClinicalTrials.gov v2.

    Uses ``aggFilters`` to restrict to the requested phases and (optionally) to
    studies that have posted results. Paginates via ``nextPageToken``.
    """
    sess = session or requests.Session()
    agg = []
    if phases:
        # aggFilters ORs values within a facet using spaces, e.g. "phase:2 3".
        agg.append("phase:" + " ".join(phases))
    if with_results:
        agg.append("results:with")
    params = {
        "query.cond": condition,
        "pageSize": min(page_size, max_studies),
    }
    if agg:
        params["aggFilters"] = ",".join(agg)

    studies: list[dict] = []
    token = None
    while len(studies) < max_studies:
        if token:
            params["pageToken"] = token
        resp = sess.get(API, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        studies.extend(payload.get("studies", []))
        token = payload.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)  # be polite to the public API
    return studies[:max_studies]


def _to_days(value: float, unit: str) -> float | None:
    if value is None:
        return None
    u = (unit or "").lower()
    if "month" in u:
        return value * MONTHS_TO_DAYS
    if "week" in u:
        return value * WEEKS_TO_DAYS
    if "day" in u:
        return value
    if "year" in u:
        return value * 365.25
    return None  # non-time unit (e.g. percentage) -> not a survival median


def _classify_endpoint(title: str) -> str | None:
    t = title or ""
    # Check OS first only when PFS isn't also present in the title.
    if _PFS.search(t):
        return "PFS"
    if _OS.search(t):
        return "OS"
    return None


def parse_outcome_medians(study: dict) -> list[dict]:
    """Extract per-arm median PFS/OS rows from one study's results section."""
    proto = study.get("protocolSection", {})
    nct = proto.get("identificationModule", {}).get("nctId")
    phase = "|".join(proto.get("designModule", {}).get("phases", []) or [])
    enrollment = (
        proto.get("designModule", {})
        .get("enrollmentInfo", {})
        .get("count")
    )
    results = study.get("resultsSection", {})
    oms = results.get("outcomeMeasuresModule", {}).get("outcomeMeasures", []) or []

    rows: list[dict] = []
    for om in oms:
        if om.get("paramType") != "MEDIAN":
            continue
        endpoint = _classify_endpoint(om.get("title", ""))
        if endpoint is None:
            continue
        unit = om.get("unitOfMeasure", "")
        groups = {g.get("id"): g.get("title") for g in om.get("groups", [])}
        for cls in om.get("classes", []):
            for cat in cls.get("categories", []):
                for meas in cat.get("measurements", []):
                    val = meas.get("value")
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    days = _to_days(val, unit)
                    if days is None:
                        continue
                    rows.append(
                        {
                            "nct_id": nct,
                            "phase": phase,
                            "enrollment": enrollment,
                            "endpoint": endpoint,
                            "arm": groups.get(meas.get("groupId"), meas.get("groupId")),
                            "median_value": val,
                            "unit": unit,
                            "median_days": round(days, 1),
                            "median_months": round(days / MONTHS_TO_DAYS, 2),
                        }
                    )
    return rows


def build_benchmark_table(studies: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for s in studies:
        rows.extend(parse_outcome_medians(s))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["endpoint", "median_days"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of historical median PFS/OS across arms (in months)."""
    if df.empty:
        return df
    return (
        df.groupby("endpoint")["median_months"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .round(2)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pull aggregate NSCLC trial benchmarks from ClinicalTrials.gov.")
    p.add_argument("--condition", default="non-small cell lung cancer")
    p.add_argument("--phases", default="2,3", help="comma-separated, e.g. 2,3")
    p.add_argument("--max-studies", type=int, default=60)
    p.add_argument(
        "--out",
        default="data/benchmarks/nsclc_trial_benchmarks.csv",
        help="output CSV path (git-ignored by default)",
    )
    args = p.parse_args(argv)

    phases = tuple(x.strip() for x in args.phases.split(",") if x.strip())
    print(f"Querying ClinicalTrials.gov for '{args.condition}' (phases {phases}, with results)...")
    studies = fetch_studies(args.condition, phases=phases, max_studies=args.max_studies)
    print(f"  fetched {len(studies)} studies")
    df = build_benchmark_table(studies)
    if df.empty:
        print("No parseable median PFS/OS outcomes found.")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  wrote {len(df)} benchmark rows -> {out}")
    print("\nHistorical median (months) across arms:")
    print(summarize(df).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
