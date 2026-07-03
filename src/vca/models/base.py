"""The ``TrajectoryModel`` interface and the ``SimulationResult`` container.

Any generative model — the dependency-light baseline, the Bayesian TGI-survival
model, or a future conditional VAE / diffusion model over trajectories — plugs
into the rest of the project by implementing two methods:

    model.fit(trial_data) -> self
    model.simulate(covariates, n_draws, times) -> SimulationResult

Everything downstream (validation, the Phase 3 power-analysis wrapper) speaks
only to this interface and to :class:`SimulationResult`, never to a model's
internals. That is what lets a more complex model be swapped in without
rewriting the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vca.data_processing.schema import TrialData

# Endpoints understood across the codebase.
ENDPOINTS = ("pfs", "os")


@dataclass
class SimulationResult:
    """Posterior/predictive draws of virtual-patient trajectories and outcomes.

    All arrays are indexed ``[draw, patient]`` (event outcomes) or
    ``[draw, patient, time]`` (trajectories). ``n_draws`` captures the model's
    full predictive uncertainty (parameter + outcome noise for a Bayesian model;
    resampling variability for the baseline).

    Event times are *latent* times to the event; administrative censoring is not
    applied here (that is a property of an observation process, not the patient).
    A draw's ``*_event`` flag is 0 only if the latent time exceeded the
    simulation horizon ``max_time``. Validation code applies censoring explicitly
    where a fair comparison to observed data requires it.
    """

    covariates: pd.DataFrame
    times: np.ndarray                       # (T,) evaluation grid, days
    sld: np.ndarray                         # (n_draws, n_patients, T), mm
    pfs_time: np.ndarray                    # (n_draws, n_patients), days
    pfs_event: np.ndarray                   # (n_draws, n_patients), {0,1}
    os_time: np.ndarray                     # (n_draws, n_patients), days
    os_event: np.ndarray                    # (n_draws, n_patients), {0,1}
    max_time: float = np.inf
    meta: dict = field(default_factory=dict)

    # -- shape helpers -------------------------------------------------------
    @property
    def n_draws(self) -> int:
        return self.pfs_time.shape[0]

    @property
    def n_patients(self) -> int:
        return self.pfs_time.shape[1]

    def _event_arrays(self, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
        if endpoint == "pfs":
            return self.pfs_time, self.pfs_event
        if endpoint == "os":
            return self.os_time, self.os_event
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {ENDPOINTS}")

    # -- predictive summaries ------------------------------------------------
    def predicted_event_prob(self, endpoint: str, t: float) -> np.ndarray:
        """P(event by time ``t``) per patient, averaged over draws -> (n_patients,)."""
        time, event = self._event_arrays(endpoint)
        occurred = (time <= t) & (event == 1)
        return occurred.mean(axis=0)

    def predicted_survival(self, endpoint: str, t: float) -> np.ndarray:
        """S(t) = 1 - P(event by t), per patient -> (n_patients,)."""
        return 1.0 - self.predicted_event_prob(endpoint, t)

    def sample_one_per_patient(
        self, endpoint: str, *, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """One latent (time, event) per patient by picking a random draw each.

        Produces a single synthetic cohort the same size as ``covariates`` — the
        object a Kaplan-Meier overlay or a simulated power analysis consumes.
        """
        rng = np.random.default_rng(seed)
        time, event = self._event_arrays(endpoint)
        idx = rng.integers(0, self.n_draws, size=self.n_patients)
        cols = np.arange(self.n_patients)
        return time[idx, cols], event[idx, cols]

    def sld_quantiles(self, q=(0.05, 0.5, 0.95)) -> dict[float, np.ndarray]:
        """Pointwise SLD quantiles over draws -> {q: (n_patients, T)}."""
        return {qq: np.quantile(self.sld, qq, axis=0) for qq in q}

    def to_events_frame(self, *, seed: int = 0) -> pd.DataFrame:
        """One simulated cohort as a canonical events-style DataFrame."""
        pfs_t, pfs_e = self.sample_one_per_patient("pfs", seed=seed)
        os_t, os_e = self.sample_one_per_patient("os", seed=seed + 1)
        return pd.DataFrame(
            {
                "patient_id": np.asarray(self.covariates.index).astype(str),
                "pfs_time_days": pfs_t,
                "pfs_event": pfs_e.astype(int),
                "os_time_days": os_t,
                "os_event": os_e.astype(int),
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SimulationResult(n_draws={self.n_draws}, n_patients={self.n_patients}, "
            f"n_times={len(self.times)})"
        )


class TrajectoryModel(ABC):
    """Abstract base class for virtual-control-arm generative models.

    Subclasses must implement :meth:`fit` and :meth:`simulate`. They should also
    set :attr:`is_fitted` and declare :attr:`required_covariates` (a subset of
    the canonical baseline columns they consume) so the pipeline can fail fast on
    mismatched inputs.
    """

    #: Baseline covariate columns the model needs present in ``simulate`` input.
    required_covariates: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, data: TrialData) -> TrajectoryModel:
        """Fit the model to canonical :class:`TrialData`. Returns ``self``."""

    @abstractmethod
    def simulate(
        self,
        covariates: pd.DataFrame,
        *,
        n_draws: int = 200,
        times: np.ndarray | None = None,
        max_time: float | None = None,
        seed: int = 0,
    ) -> SimulationResult:
        """Generate predictive trajectories/outcomes for the given covariates.

        Parameters
        ----------
        covariates
            One row per virtual patient; must contain
            :attr:`required_covariates`. The DataFrame index is used as the
            patient key in the returned :class:`SimulationResult`.
        n_draws
            Number of predictive draws per patient.
        times
            Day grid on which to evaluate SLD trajectories. Defaults to a
            model-chosen grid if ``None``.
        max_time
            Simulation horizon for latent event times (days).
        seed
            RNG seed for reproducibility.
        """

    # -- shared utilities ----------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be .fit() before .simulate()")

    def _check_covariates(self, covariates: pd.DataFrame) -> None:
        missing = [c for c in self.required_covariates if c not in covariates.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.simulate() missing required covariates: {missing}"
            )

    @staticmethod
    def _default_time_grid(max_time: float = 720.0, step: float = 30.0) -> np.ndarray:
        return np.arange(0.0, max_time + step, step)
