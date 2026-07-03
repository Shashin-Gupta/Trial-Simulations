"""Dependency-light baseline model: within-stratum Kaplan-Meier resampling.

This is a transparent nonparametric external-control baseline. It requires only
numpy/pandas/scipy (no numpyro/JAX), so the whole pipeline runs anywhere, and it
is a genuine comparator: a more sophisticated model has to *beat* it on the
held-out validation metrics to justify its complexity.

How it works
------------
Patients are grouped into coarse covariate strata (age band x ECOG x stage
group x histology). To simulate outcomes for a query patient we:

- find the finest stratum with at least ``min_donors`` donors (backing off to
  coarser strata, ultimately the marginal pool, when data are sparse);
- draw latent PFS/OS times by inverting the stratum's Kaplan-Meier curve
  (:mod:`vca._km`), which correctly turns right-censored donor follow-up into a
  survival distribution;
- draw an SLD trajectory by resampling a donor's observed tumour-size path and
  interpolating it onto the requested time grid.

Limitations (called out for honesty; see docs/methodology.md)
-------------------------------------------------------------
- It cannot extrapolate beyond the covariate combinations present in the data;
  a query far from the training support falls back to a coarse/marginal stratum.
- Strata are hand-chosen and coarse; it ignores within-stratum covariate
  variation. The Bayesian TGI-survival model is the intended improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vca._km import kaplan_meier, sample_from_km
from vca.data_processing.schema import TrialData
from vca.models.base import SimulationResult, TrajectoryModel

# Backoff ladder: finest stratum first, marginal pool last. Each entry lists the
# bucket functions whose outputs form the stratum key at that level.
_STRATIFIERS = ("age_band", "ecog_bucket", "stage_group", "histology")


def _age_band(age) -> str:
    if pd.isna(age):
        return "unk"
    if age < 60:
        return "<60"
    if age < 70:
        return "60-69"
    return "70+"


def _ecog_bucket(ecog) -> str:
    if pd.isna(ecog):
        return "unk"
    e = int(ecog)
    return "0" if e == 0 else ("1" if e == 1 else "2+")


def _stage_group(stage) -> str:
    if not isinstance(stage, str):
        return "unk"
    return "IV" if stage.upper().startswith("IV") else "III/other"


def _histology(h) -> str:
    return h if isinstance(h, str) and h else "unk"


def _bucket_row(row) -> dict[str, str]:
    return {
        "age_band": _age_band(row.get("age")),
        "ecog_bucket": _ecog_bucket(row.get("ecog_ps")),
        "stage_group": _stage_group(row.get("stage")),
        "histology": _histology(row.get("histology")),
    }


# Coarsening order: drop age first, then stage, then histology, then ECOG.
_BACKOFF_LEVELS = [
    ("age_band", "ecog_bucket", "stage_group", "histology"),
    ("ecog_bucket", "stage_group", "histology"),
    ("ecog_bucket", "histology"),
    ("ecog_bucket",),
    (),  # marginal
]


class MarginalResamplingModel(TrajectoryModel):
    """Within-stratum Kaplan-Meier resampling baseline.

    Parameters
    ----------
    min_donors
        Minimum donors required to use a stratum before backing off to a coarser
        one.
    """

    required_covariates = ("age", "ecog_ps", "stage", "histology")

    def __init__(self, min_donors: int = 8) -> None:
        super().__init__()
        self.min_donors = min_donors
        self._baseline: pd.DataFrame | None = None
        self._events: pd.DataFrame | None = None
        self._buckets: pd.DataFrame | None = None
        self._donor_paths: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._km_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    # -- fitting -------------------------------------------------------------
    def fit(self, data: TrialData) -> MarginalResamplingModel:
        data.validate(strict=False)
        self._baseline = data.baseline.set_index("patient_id")
        self._events = data.events.set_index("patient_id")
        self._buckets = pd.DataFrame(
            [_bucket_row(r) for _, r in self._baseline.reset_index().iterrows()],
            index=self._baseline.index,
        )
        # Donor SLD paths keyed by patient_id.
        self._donor_paths = {}
        for pid, grp in data.longitudinal.groupby("patient_id"):
            g = grp.sort_values("time_days")
            self._donor_paths[str(pid)] = (
                g["time_days"].to_numpy(float),
                g["sld_mm"].to_numpy(float),
            )
        self._km_cache = {}
        self.is_fitted = True
        return self

    # -- donor selection -----------------------------------------------------
    def _donor_pool(self, buckets: dict[str, str]) -> tuple[np.ndarray, tuple]:
        """Return donor patient_ids and the stratum key actually used."""
        assert self._buckets is not None
        for level in _BACKOFF_LEVELS:
            if level:
                mask = np.ones(len(self._buckets), dtype=bool)
                for col in level:
                    mask &= (self._buckets[col].to_numpy() == buckets[col])
                pool = self._buckets.index[mask].to_numpy()
                key = tuple((col, buckets[col]) for col in level)
            else:
                pool = self._buckets.index.to_numpy()
                key = ()
            if len(pool) >= self.min_donors:
                return pool, key
        return self._buckets.index.to_numpy(), ()  # marginal fallback

    def _stratum_km(self, key: tuple, pool: np.ndarray, endpoint: str):
        cache_key = (key, endpoint)
        if cache_key not in self._km_cache:
            ev = self._events.loc[pool]
            t = ev[f"{endpoint}_time_days"].to_numpy(float)
            e = ev[f"{endpoint}_event"].to_numpy(int)
            self._km_cache[cache_key] = kaplan_meier(t, e)
        return self._km_cache[cache_key]

    # -- simulation ----------------------------------------------------------
    def simulate(
        self,
        covariates: pd.DataFrame,
        *,
        n_draws: int = 200,
        times: np.ndarray | None = None,
        max_time: float | None = None,
        seed: int = 0,
    ) -> SimulationResult:
        self._check_fitted()
        self._check_covariates(covariates)
        rng = np.random.default_rng(seed)
        if times is None:
            times = self._default_time_grid()
        times = np.asarray(times, float)
        if max_time is None:
            max_time = 5 * 365.0
        n_pat, T = len(covariates), len(times)

        sld = np.empty((n_draws, n_pat, T))
        pfs_t = np.empty((n_draws, n_pat))
        pfs_e = np.empty((n_draws, n_pat), dtype=int)
        os_t = np.empty((n_draws, n_pat))
        os_e = np.empty((n_draws, n_pat), dtype=int)

        for j, (_, row) in enumerate(covariates.iterrows()):
            buckets = _bucket_row(row)
            pool, key = self._donor_pool(buckets)

            for endpoint, out_t, out_e in (("pfs", pfs_t, pfs_e), ("os", os_t, os_e)):
                et, sv = self._stratum_km(key, pool, endpoint)
                tt, ee = sample_from_km(et, sv, n_draws, rng, max_time)
                out_t[:, j] = tt
                out_e[:, j] = ee

            # SLD trajectories: resample donor paths from the pool.
            donor_ids = rng.choice(pool, size=n_draws)
            for d, did in enumerate(donor_ids):
                dt, dy = self._donor_paths.get(str(did), (np.array([0.0]), np.array([np.nan])))
                sld[d, j, :] = np.interp(times, dt, dy, left=dy[0], right=dy[-1])

        return SimulationResult(
            covariates=covariates.copy(),
            times=times,
            sld=sld,
            pfs_time=pfs_t,
            pfs_event=pfs_e,
            os_time=os_t,
            os_event=os_e,
            max_time=float(max_time),
            meta={"model": "MarginalResamplingModel", "min_donors": self.min_donors},
        )
