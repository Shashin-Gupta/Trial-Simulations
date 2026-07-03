"""Phase 3 (STUB): simulated power / feasibility analysis for a trial design.

# TODO: Phase 3 — DO NOT IMPLEMENT until Phases 0-2 are validated on REAL
# Project Data Sphere data. Building a power-analysis tool on an unvalidated
# generator would produce confident, wrong feasibility numbers.

Intended design
---------------
Given a fitted, *validated* ``TrajectoryModel`` and a proposed trial design, this
module will estimate operating characteristics (power, type-I error, expected
events, feasibility) by Monte Carlo:

    design = TrialDesign(
        inclusion=InclusionCriteria(age=(18, 75), ecog_max=1, stage={"IV"}),
        n_control=100, n_treatment=100,
        endpoint="pfs", analysis="logrank",
        assumed_hazard_ratio=0.7, follow_up_days=540, accrual_days=360,
    )
    result = simulate_power(model, design, n_trials=1000)
    # result.power, result.expected_events, result.type_i_error, ...

Sketch of the procedure (per simulated trial):
  1. Sample a control cohort's covariates consistent with ``inclusion`` (from the
     empirical covariate distribution of the fitted data, filtered to criteria).
  2. ``model.simulate(...)`` -> virtual control-arm outcomes.
  3. Generate treatment-arm outcomes under the design's assumed effect (e.g. by
     scaling the control hazard by ``assumed_hazard_ratio``).
  4. Apply accrual + administrative censoring for ``follow_up_days``.
  5. Run the planned analysis (log-rank / Cox) and record rejection of H0.
  Repeat; power = rejection rate under the alternative, type-I error under HR=1.

Open validity questions to resolve before shipping (see docs/methodology.md):
  - Using a virtual (non-concurrent) control inflates type-I error if the model
    is even slightly miscalibrated; quantify this on real data first.
  - Covariate shift between the historical fit population and the proposed trial
    population must be checked and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InclusionCriteria:
    """Eligibility filter for the simulated cohort. # TODO: Phase 3"""

    age: tuple[float, float] = (18.0, 120.0)
    ecog_max: int = 2
    stage: set[str] = field(default_factory=lambda: {"IIIB", "IV"})
    histology: set[str] | None = None


@dataclass
class TrialDesign:
    """A proposed trial design to evaluate. # TODO: Phase 3"""

    inclusion: InclusionCriteria
    n_control: int
    n_treatment: int
    endpoint: str = "pfs"
    analysis: str = "logrank"
    assumed_hazard_ratio: float = 0.7
    follow_up_days: float = 540.0
    accrual_days: float = 360.0
    alpha: float = 0.05


@dataclass
class PowerResult:
    """Operating characteristics from the Monte Carlo. # TODO: Phase 3"""

    power: float
    type_i_error: float
    expected_events: float
    n_trials: int


def simulate_power(model, design: TrialDesign, n_trials: int = 1000) -> PowerResult:
    """Estimate power/feasibility for ``design`` under ``model``. # TODO: Phase 3.

    Not implemented: gated on real-data validation (Phases 0-2). See module
    docstring for the intended Monte Carlo procedure.
    """
    raise NotImplementedError(
        "Phase 3 power analysis is intentionally not implemented until the "
        "generator is validated on real Project Data Sphere data. See "
        "docs/methodology.md and the module docstring for the planned design."
    )
