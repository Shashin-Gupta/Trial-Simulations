"""Hierarchical Bayesian tumour-growth-inhibition + survival joint model.

This is the project's primary scientific model (Phase 1). It is a joint model in
the pharmacometric TGI-OS tradition (Stein; Claret; Bruno et al.): a longitudinal
sub-model for RECIST tumour size and a time-to-event sub-model that share a
latent per-patient tumour *growth rate*, so that the two are estimated coherently.

Longitudinal sub-model (Stein bi-exponential TGI)
-------------------------------------------------
    SLD_i(t) = y0_i * ( exp(-d_i * t) + exp(g_i * t) - 1 )
    log SLD_obs_ij ~ Normal( log SLD_i(t_ij), sigma_obs )

``y0_i`` is anchored to the measured baseline SLD. The per-patient shrinkage
rate ``d_i`` and regrowth rate ``g_i`` are hierarchical, with population means
that depend on baseline covariates ``X_i``:

    log d_i = mu_d + X_i . beta_d + sigma_d * z^d_i        (non-centred)
    log g_i = mu_g + X_i . beta_g + sigma_g * z^g_i

Time-to-event sub-model (Weibull PH, coupled to growth)
-------------------------------------------------------
    scale^E_i = exp( a_E + X_i . gamma_E + theta_E * (log g_i - mu_g) )
    T^E_i ~ Weibull(scale^E_i, k_E)          for E in {PFS, OS}

with right-censoring handled by the Weibull survival function. ``theta_E`` is the
coupling: a positive latent growth rate shortens survival. Sharing ``log g_i``
across the longitudinal and survival likelihoods is what makes this a *joint*
model rather than two independent regressions.

The ``numpyro``/``jax`` dependency is imported lazily so the rest of the package
works without it (install with ``pip install -e ".[bayes]"``).

Modelling assumptions flagged for validity (see docs/methodology.md)
--------------------------------------------------------------------
- Baseline SLD is treated as a known anchor ``y0_i`` (ignores its measurement
  error).
- PFS and OS are modelled as separate Weibulls sharing the growth link; the
  model does not enforce OS >= PFS (no explicit semi-competing-risks structure).
- Non-informative (independent) censoring is assumed.
- The growth link is linear in ``log g_i``; alternative tumour-dynamic summaries
  (e.g. week-8 tumour ratio) are a planned sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vca.data_processing.schema import TrialData
from vca.models.base import SimulationResult, TrajectoryModel
from vca.models.preprocessing import CovariatePreprocessor


def _lazy_imports():
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TGISurvivalModel needs the [bayes] extra. Install with:\n"
            '    pip install -e ".[bayes]"\n'
            "or, if JAX has no wheel for your Python version, use Python 3.11/3.12."
        ) from exc
    return jax, jnp, numpyro, dist


def _numpyro_model(X, y0, patient_idx, obs_time, log_sld, pfs_t, pfs_e, os_t, os_e):
    """NumPyro model function. All arrays are jax/np arrays.

    ``X``          : (N, P) design matrix, one row per patient.
    ``y0``         : (N,) baseline SLD anchor.
    ``patient_idx``: (M,) patient index for each of M longitudinal measurements.
    ``obs_time``   : (M,) measurement times (days).
    ``log_sld``    : (M,) log observed SLD.
    ``pfs_*``/``os_*`` : (N,) event times and indicators.

    ``numpyro`` and friends are imported inside because NumPyro's MCMC forwards
    only the data arguments to the model.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    N, P = X.shape

    # --- population priors --------------------------------------------------
    mu_d = numpyro.sample("mu_d", dist.Normal(jnp.log(0.004), 0.6))
    mu_g = numpyro.sample("mu_g", dist.Normal(jnp.log(0.002), 0.6))
    beta_d = numpyro.sample("beta_d", dist.Normal(jnp.zeros(P), 0.3))
    beta_g = numpyro.sample("beta_g", dist.Normal(jnp.zeros(P), 0.3))
    sigma_d = numpyro.sample("sigma_d", dist.HalfNormal(0.5))
    sigma_g = numpyro.sample("sigma_g", dist.HalfNormal(0.5))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.25))

    # --- per-patient latent rates (non-centred) -----------------------------
    with numpyro.plate("patients", N):
        z_d = numpyro.sample("z_d", dist.Normal(0.0, 1.0))
        z_g = numpyro.sample("z_g", dist.Normal(0.0, 1.0))
    log_d = mu_d + X @ beta_d + sigma_d * z_d
    log_g = mu_g + X @ beta_g + sigma_g * z_g
    d = jnp.exp(log_d)
    g = jnp.exp(log_g)

    # --- longitudinal likelihood -------------------------------------------
    # Exponent clipping keeps the mean finite while NUTS explores extreme draws
    # (e.g. a large growth rate over ~2 years would otherwise overflow exp()).
    di = d[patient_idx]
    gi = g[patient_idx]
    y0i = y0[patient_idx]
    grow = jnp.exp(jnp.clip(gi * obs_time, None, 15.0))
    shrink = jnp.exp(jnp.clip(-di * obs_time, -30.0, 0.0))
    mean_sld = jnp.clip(y0i * (shrink + grow - 1.0), 1e-3, 5e4)
    numpyro.sample("sld_obs", dist.Normal(jnp.log(mean_sld), sigma_obs), obs=log_sld)

    # --- survival sub-models (Weibull PH, coupled to growth) ----------------
    growth_dev = log_g - mu_g  # centred growth deviation shared across endpoints
    for name, gt, ge, a_loc, k_loc in (
        ("pfs", pfs_t, pfs_e, jnp.log(150.0), 1.2),
        ("os", os_t, os_e, jnp.log(300.0), 1.2),
    ):
        a = numpyro.sample(f"a_{name}", dist.Normal(a_loc, 1.0))
        gamma = numpyro.sample(f"gamma_{name}", dist.Normal(jnp.zeros(P), 0.4))
        theta = numpyro.sample(f"theta_{name}", dist.Normal(0.0, 1.0))
        k = numpyro.sample(f"k_{name}", dist.LogNormal(jnp.log(k_loc), 0.3))
        # Clip the linear predictor so the Weibull scale stays in a sane,
        # finite range (~1 day .. ~1.2M days) during exploration.
        scale = jnp.exp(jnp.clip(a + X @ gamma + theta * growth_dev, 0.0, 14.0))
        # Weibull log-likelihood with right-censoring.
        wb = dist.Weibull(scale=scale, concentration=k)
        log_surv = -((gt / scale) ** k)          # log S(t)
        ll = jnp.where(ge == 1, wb.log_prob(gt), log_surv)
        numpyro.factor(f"{name}_ll", ll.sum())


class TGISurvivalModel(TrajectoryModel):
    """Hierarchical Bayesian TGI + Weibull-survival joint model (NumPyro/NUTS).

    Parameters
    ----------
    num_warmup, num_samples, num_chains
        NUTS settings. Defaults are modest; increase for publication-quality
        posteriors and check diagnostics (see :meth:`diagnostics`).
    seed
        PRNG seed for MCMC.
    """

    required_covariates = ("baseline_sld_mm",)

    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.seed = seed
        self.preprocessor = CovariatePreprocessor()
        self._posterior: dict[str, np.ndarray] | None = None
        self._mcmc = None

    # -- fitting -------------------------------------------------------------
    def fit(self, data: TrialData) -> "TGISurvivalModel":
        data.validate(strict=False)
        jax, jnp, numpyro, dist = _lazy_imports()
        from numpyro.infer import MCMC, NUTS

        base = data.baseline.reset_index(drop=True)
        if "baseline_sld_mm" not in base:
            raise ValueError("TGISurvivalModel requires 'baseline_sld_mm' in baseline.")

        # Design matrix and baseline anchor.
        X = self.preprocessor.fit_transform(base)
        y0 = base["baseline_sld_mm"].to_numpy(float)
        pid_to_idx = {str(p): i for i, p in enumerate(base["patient_id"])}

        lg = data.longitudinal.copy()
        lg = lg[lg["patient_id"].astype(str).isin(pid_to_idx)]
        patient_idx = lg["patient_id"].astype(str).map(pid_to_idx).to_numpy()
        obs_time = lg["time_days"].to_numpy(float)
        log_sld = np.log(np.clip(lg["sld_mm"].to_numpy(float), 1e-3, None))

        ev = data.events.set_index("patient_id").loc[base["patient_id"]]
        pfs_t = np.clip(ev["pfs_time_days"].to_numpy(float), 1.0, None)
        pfs_e = ev["pfs_event"].to_numpy(int)
        os_t = np.clip(ev["os_time_days"].to_numpy(float), 1.0, None)
        os_e = ev["os_event"].to_numpy(int)

        kernel = NUTS(_numpyro_model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False,
            chain_method="sequential",
        )
        mcmc.run(
            jax.random.PRNGKey(self.seed),
            jnp.asarray(X), jnp.asarray(y0), jnp.asarray(patient_idx),
            jnp.asarray(obs_time), jnp.asarray(log_sld),
            jnp.asarray(pfs_t), jnp.asarray(pfs_e),
            jnp.asarray(os_t), jnp.asarray(os_e),
        )
        self._mcmc = mcmc
        self._posterior = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self.is_fitted = True
        return self

    def diagnostics(self):  # pragma: no cover - requires fitted model
        """Print NUTS convergence diagnostics (R-hat, ESS)."""
        self._check_fitted()
        self._mcmc.print_summary(exclude_deterministic=True)

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

        post = self._posterior
        S = post["mu_d"].shape[0]
        s = rng.integers(0, S, size=n_draws)  # posterior sample per draw

        Xn = self.preprocessor.transform(covariates)          # (n_pat, P)
        y0 = covariates["baseline_sld_mm"].to_numpy(float)     # (n_pat,)

        # Population-predictive latent rates for these (new) patients.
        lin_d = post["mu_d"][s][:, None] + np.einsum("dp,np->dn", post["beta_d"][s], Xn)
        lin_g = post["mu_g"][s][:, None] + np.einsum("dp,np->dn", post["beta_g"][s], Xn)
        z_d = rng.standard_normal((n_draws, n_pat))
        z_g = rng.standard_normal((n_draws, n_pat))
        log_d = lin_d + post["sigma_d"][s][:, None] * z_d
        log_g = lin_g + post["sigma_g"][s][:, None] * z_g
        d = np.exp(log_d)                                      # (n_draws, n_pat)
        g = np.exp(log_g)

        # Trajectories with multiplicative observation noise.
        tt = times[None, None, :]
        mean_sld = y0[None, :, None] * (
            np.exp(-d[:, :, None] * tt) + np.exp(g[:, :, None] * tt) - 1.0
        )
        mean_sld = np.clip(mean_sld, 1e-3, None)
        noise = rng.standard_normal((n_draws, n_pat, T)) * post["sigma_obs"][s][:, None, None]
        sld = np.clip(mean_sld * np.exp(noise), 1e-3, None)

        # Survival draws.
        growth_dev = log_g - post["mu_g"][s][:, None]
        pfs_t, pfs_e = self._draw_weibull("pfs", s, Xn, growth_dev, n_draws, rng, max_time)
        os_t, os_e = self._draw_weibull("os", s, Xn, growth_dev, n_draws, rng, max_time)

        return SimulationResult(
            covariates=covariates.copy(),
            times=times,
            sld=sld,
            pfs_time=pfs_t,
            pfs_event=pfs_e,
            os_time=os_t,
            os_event=os_e,
            max_time=float(max_time),
            meta={"model": "TGISurvivalModel", "n_posterior_samples": int(S)},
        )

    def _draw_weibull(self, name, s, Xn, growth_dev, n_draws, rng, max_time):
        post = self._posterior
        scale = np.exp(
            post[f"a_{name}"][s][:, None]
            + np.einsum("dp,np->dn", post[f"gamma_{name}"][s], Xn)
            + post[f"theta_{name}"][s][:, None] * growth_dev
        )
        k = post[f"k_{name}"][s][:, None]
        u = rng.random(scale.shape)
        t = scale * (-np.log(u)) ** (1.0 / k)   # inverse-CDF Weibull sampling
        event = (t <= max_time).astype(int)
        t = np.minimum(t, max_time)
        return t, event
