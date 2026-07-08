"""Step 1: profile all five real PDS trials and write results/profiling/*.

Usage:  python scripts/profile_trials.py
Outputs (git-ignored; may reference real patients):
  results/profiling/<trial>_profile.json   per-trial structured profile
  results/profiling/summary.csv             one-row-per-trial headline table
  results/profiling/summary.md              human-readable summary + flags
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from vca.data_processing.pds_trials import load_all
from vca.validation.profiling import profile_trial, profiles_to_frame

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results" / "profiling"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = load_all()
    profiles = {}
    for tid, rt in trials.items():
        p = profile_trial(rt)
        profiles[tid] = p
        (OUT / f"{tid}_profile.json").write_text(json.dumps(p, indent=2))

    table = profiles_to_frame(profiles)
    table.to_csv(OUT / "summary.csv", index=False)

    lines = ["# Real-data profiling summary (Step 1)", "",
             "One-row-per-trial headline metrics; see `<trial>_profile.json` for full "
             "covariate distributions and missingness.", "",
             table.to_markdown(index=False), "",
             "## Censoring / endpoint conventions", ""]
    for tid, p in profiles.items():
        lines += [f"- **{tid}** ({p['regimen']}, {p['line']}, {p['histology_label']}): "
                  f"PFS — {p['pfs_convention']}; OS — {p['os_convention']}."]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(table.to_string(index=False))
    print(f"\nWrote profiles to {OUT}")


if __name__ == "__main__":
    main()
