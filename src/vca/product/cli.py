"""Phase 3 (STUB): command-line entry point for the trial-design tool.

# TODO: Phase 3 — scaffold only. Wire this up after real-data validation.

Planned usage:

    vca-power --config my_trial.yaml            # -> prints power / feasibility
    vca-power --n-control 100 --hr 0.7 --endpoint pfs --follow-up 540

It will load a fitted+validated model artifact, build a TrialDesign from the
CLI/config, call vca.product.power.simulate_power, and print operating
characteristics. See vca.product.power for the intended design.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - stub
    # TODO: Phase 3 — parse args/config, load validated model, build TrialDesign,
    # call simulate_power, render a report.
    raise NotImplementedError(
        "Phase 3 CLI is a stub. Complete Phases 0-2 validation on real data first."
    )


if __name__ == "__main__":
    raise SystemExit(main())
