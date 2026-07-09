#!/usr/bin/env python
"""Fig. 4 (supporting): immortal-time bias in the subsequent-therapy analysis
(trial 108). Grouped bar of median OS by subsequent-therapy status under a naive
ever/never split vs a 180-day landmark, showing the direction reversal that
supports excluding subsequent therapy as the cause of OS under-prediction
(paper.md Fig. 4; methodology.md §4.2).

This is a *presentation* figure, not a pipeline artifact. The naive medians are
read from the committed diagnostic CSV; the landmark medians are the documented
trial-108 result from scripts/diagnose_os_subsequent_therapy.py, which recomputes
them from the DUA-protected patient-level export (not committed, so not re-run
here). Both numbers are stated in methodology.md §4.2.

    python scripts/plot_fig4_landmark_reversal.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAG_CSV = ROOT / "results" / "real_data" / "subsequent_therapy_diagnostic.csv"
OUT = ROOT / "results" / "real_data" / "figures" / "108_subsequent_therapy_landmark.png"

# Okabe-Ito colorblind-safe pair (validated: CVD ΔE >> 12); colour follows the
# category, not the group, so the reversal is visible as blue tall -> blue short.
C_THERAPY = "#0072B2"  # subsequent therapy: received / started by 180 d
C_NONE = "#E69F00"     # no subsequent therapy: never / not started by 180 d
INK = "#222222"

# Naive ever/never split: read from the committed diagnostic (trial 108 row).
_diag = pd.read_csv(DIAG_CSV).set_index("trial")
naive_therapy = float(_diag.loc[108, "os_med_subseq_mo"])  # 15.8
naive_none = float(_diag.loc[108, "os_med_none_mo"])       # 6.9

# 180-day landmark (alive at 180 d; classified by therapy started by then):
# documented trial-108 result (methodology.md §4.2). Not in the committed CSV
# because it is recomputed from the DUA-protected patient-level export.
landmark_therapy = 12.3
landmark_none = 17.7


def main() -> None:
    groups = ["Naive split\n(ever vs never)", "180-day landmark\n(alive at 180 d)"]
    therapy_vals = [naive_therapy, landmark_therapy]
    none_vals = [naive_none, landmark_none]

    x = range(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    bars_t = ax.bar([i - w / 2 for i in x], therapy_vals, w,
                    label="Subsequent therapy", color=C_THERAPY,
                    edgecolor="white", linewidth=1.2)
    bars_n = ax.bar([i + w / 2 for i in x], none_vals, w,
                    label="No subsequent therapy", color=C_NONE,
                    edgecolor="white", linewidth=1.2)

    # Direct value labels (relief for the low-contrast fill; text in ink, not the
    # series colour).
    for bars in (bars_t, bars_n):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color=INK)

    ax.set_ylabel("Median overall survival (months)", fontsize=11, color=INK)
    ax.set_title("Trial 108: subsequent therapy vs OS — naive split reverses "
                 "under a landmark", fontsize=11, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=10, color=INK)
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}  "
          f"(naive {naive_therapy:.1f}/{naive_none:.1f}; "
          f"landmark {landmark_therapy:.1f}/{landmark_none:.1f})")


if __name__ == "__main__":
    main()
