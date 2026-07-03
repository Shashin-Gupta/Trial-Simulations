"""Phase 3 (STUB): Streamlit app for the trial-design / power tool.

# TODO: Phase 3 — scaffold only. Do NOT build until Phases 0-2 are validated on
# real Project Data Sphere data.

Planned UX
----------
A single-page app where a user (small biotech / academic investigator) enters a
proposed trial design and gets back a simulated power/feasibility analysis:

  Sidebar inputs:
    - inclusion criteria (age range, ECOG max, stage, histology)
    - sample sizes (control / treatment), or "virtual control only" mode
    - endpoint (PFS / OS), assumed hazard ratio, follow-up + accrual duration
  Main panel outputs:
    - estimated power vs sample size curve
    - simulated KM curves for the virtual control arm (with uncertainty)
    - expected number of events, expected trial duration
    - prominent CALIBRATION/VALIDITY caveats from the model's validation report

Run (once implemented):  streamlit run src/vca/product/app.py
"""

from __future__ import annotations


def main() -> None:  # pragma: no cover - stub
    # TODO: Phase 3 — build the Streamlit UI described above, backed by
    # vca.product.power.simulate_power and a loaded validated model artifact.
    raise NotImplementedError(
        "Phase 3 Streamlit app is a stub. Validate on real data (Phases 0-2) first."
    )


if __name__ == "__main__":
    main()
