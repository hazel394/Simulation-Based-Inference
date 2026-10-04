"""
Six inference methods for the Lotka-Volterra model.

Every function returns an (n_samples, N_PARAMS) numpy array of
log-parameter samples approximating the posterior p(log theta | x_o).

Each method takes a `budget` argument giving the
total number of simulator calls it is allowed. are free):

"""

from __future__ import annotations
import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

import lotka_volterra as lv



def _safe_summary(log_theta, noise_sd, rng=None):
    """One simulation + summary; returns None on runaway."""
    try:
        states = lv.simulate_noisy_trajectory(np.exp(log_theta),
                                              noise_sd=noise_sd, rng=rng)
        return lv.calc_summary_stats(states)
    except RuntimeError:
        return None


def _safe_trajectory(log_theta, noise_sd, rng=None):
    """One simulation returning the full noisy trajectory; None on runaway."""
    try:
        return lv.simulate_noisy_trajectory(np.exp(log_theta),
                                            noise_sd=noise_sd, rng=rng)
    except RuntimeError:
        return None


def _standardize_scale(sim_summaries):
    """Per-coordinate std of the pool, floored to avoid divide-by-zero."""
    sd = sim_summaries.std(axis=0)
    return np.where(sd < 1e-8, 1e-8, sd)


# 1. ABC - Rejection
def run_abc_rejection(x_o, budget, keep_fraction=0.02,
                      noise_sd=lv.OBS_NOISE_SD, rng=None,
                      return_info=False, verbose=True):
    """
    Draw `budget` samples from the prior, simulate, keep the closest
    `keep_fraction`. Distance is Euclidean on standardized summary stats.

    Diagnostics reported (via `return_info=True` and/or `verbose=True`):
        acceptance_rate  -- equal to keep_fraction by construction; not
                            informative on its own.
        eps_effective    -- the largest kept distance (i.e. the implicit
                            epsilon threshold this rejection loop enforced).
        dist_quantiles   -- 10/50/90 percentiles of ALL prior-draw distances.
                            Compare to eps_effective: if the top 2% of
                            distances is close to the median distance, the
                            summaries are not discriminating well and the
                            posterior will look prior-flat.
    """
    rng = np.random.default_rng() if rng is None else rng

    thetas, summaries = [], []
    for _ in tqdm(range(budget), desc="ABC-Rejection", leave=False):
        lt = lv.sample_log_prior(rng=rng)
        s = _safe_summary(lt, noise_sd, rng=rng)
        if s is not None and np.all(np.isfinite(s)):
            thetas.append(lt)
            summaries.append(s)

    thetas = np.array(thetas)
    summaries = np.array(summaries)
    sd = _standardize_scale(summaries)
    dists = np.linalg.norm((summaries - x_o) / sd, axis=1)

    n_keep = max(1, int(np.ceil(keep_fraction * len(thetas))))
    keep_idx = np.argpartition(dists, n_keep)[:n_keep]
    samples = thetas[keep_idx]

    info = dict(
        acceptance_rate = keep_fraction,
        n_valid_sims    = len(thetas),
        n_kept          = n_keep,
        eps_effective   = float(dists[keep_idx].max()),
        dist_quantiles  = dict(zip([10, 50, 90],
                                   np.percentile(dists, [10, 50, 90]).round(3))),
    )
    if verbose:
        print(f"[ABC-Rej]  kept {n_keep}/{len(thetas)} "
              f"(acc={keep_fraction:.1%}), "
              f"eps_effective={info['eps_effective']:.3f}, "
              f"dist_quantiles(10/50/90)={info['dist_quantiles']}")
    return (samples, info) if return_info else samples


# 2. ABC - MCMC (random-walk Metropolis with Gaussian ABC kernel)
def run_abc_mcmc(x_o, budget, epsilon=None, proposal_sd=0.15,
                 pilot_frac=0.05, eps_quantile=0.01,
                 noise_sd=lv.OBS_NOISE_SD, rng=None,
                 return_info=False, verbose=True):
    """
    Random-walk Metropolis targeting the ABC posterior with a Gaussian
    kernel on standardized summary distance.

    Budget accounting:
        pilot phase   : pilot_frac * budget  simulator calls
                        (used to pick epsilon and the standardization scale)
        MCMC phase    : (1 - pilot_frac) * budget  simulator calls
                        (one simulator call per MCMC step)

    Epsilon selection:
        If `epsilon` is None, it is set to the `eps_quantile`-th quantile of
        pilot distances (default 1%). This is a small ABC bandwidth: the
        chain will accept aggressively when close to truth but reject
        proposals that drift into low-density regions. If the resulting
        acceptance rate is too low (< ~10%), raise eps_quantile; if the
        posterior is prior-flat, lower it.
    """
    rng = np.random.default_rng() if rng is None else rng

    # pilot: choose sd + epsilon 
    n_pilot = int(pilot_frac * budget)
    pilot_summaries = []
    for _ in range(n_pilot):
        lt = lv.sample_log_prior(rng=rng)
        s = _safe_summary(lt, noise_sd, rng=rng)
        if s is not None and np.all(np.isfinite(s)):
            pilot_summaries.append(s)
    pilot_summaries = np.array(pilot_summaries)
    sd = _standardize_scale(pilot_summaries)
    pilot_dists = np.linalg.norm((pilot_summaries - x_o) / sd, axis=1)
    if epsilon is None:
        epsilon = float(np.quantile(pilot_dists, eps_quantile))

    # ---- Gaussian ABC kernel (log) ----
    def log_kernel(dist):
        return -0.5 * (dist / epsilon) ** 2

    # ---- initialise chain from a valid starting point ----
    log_curr = lv.LOG_TRUE_PARAMS.copy()   # warm start; simplest choice
    s_curr = None
    while s_curr is None:
        s_curr = _safe_summary(log_curr, noise_sd, rng=rng)
        if s_curr is None:
            log_curr = lv.sample_log_prior(rng=rng)
    dist_curr = np.linalg.norm((s_curr - x_o) / sd)
    logk_curr = log_kernel(dist_curr)

    # MCMC loop 
    n_mcmc = budget - n_pilot
    chain = np.zeros((n_mcmc, lv.N_PARAMS))
    dist_trace = np.zeros(n_mcmc)     # distance at the current state
    n_accept = 0
    for t in tqdm(range(n_mcmc), desc="ABC-MCMC", leave=False):
        log_prop = log_curr + rng.normal(0, proposal_sd, size=lv.N_PARAMS)
        if lv.log_prior(log_prop) == -np.inf:
            chain[t]      = log_curr
            dist_trace[t] = dist_curr
            continue
        s_prop = _safe_summary(log_prop, noise_sd, rng=rng)
        if s_prop is None or not np.all(np.isfinite(s_prop)):
            chain[t]      = log_curr
            dist_trace[t] = dist_curr
            continue
        dist_prop = np.linalg.norm((s_prop - x_o) / sd)
        logk_prop = log_kernel(dist_prop)
        if np.log(rng.uniform()) < (logk_prop - logk_curr):
            log_curr, logk_curr, dist_curr = log_prop, logk_prop, dist_prop
            n_accept += 1
        chain[t]      = log_curr
        dist_trace[t] = dist_curr

    # drop burn-in (10% of the chain, capped at 2000)
    burn = min(n_mcmc // 10, 2000)
    samples = chain[burn:]

    info = dict(
        acceptance_rate = n_accept / n_mcmc,
        epsilon         = epsilon,
        n_pilot_used    = len(pilot_summaries),
        pilot_dist_quantiles = dict(zip(
            [1, 10, 50, 90],
            np.percentile(pilot_dists, [1, 10, 50, 90]).round(3))),
        chain_dist_quantiles = dict(zip(
            [10, 50, 90],
            np.percentile(dist_trace[burn:], [10, 50, 90]).round(3))),
    )
    if verbose:
        print(f"[ABC-MCMC] acc={info['acceptance_rate']:.1%}, "
              f"eps={epsilon:.3f}, "
              f"pilot d(1/10/50/90)={info['pilot_dist_quantiles']}, "
              f"chain d(10/50/90)={info['chain_dist_quantiles']}")
    return (samples, info) if return_info else samples


# 3. Wasserstein-ABC 
def _wasserstein_2d(traj_a, traj_b, scale):
    """
    2-Wasserstein cost between two trajectories viewed as empirical measures
    on R^2 with uniform weights, using per-coordinate scaling.
    """
    import ot
    a = traj_a / scale
    b = traj_b / scale
    n = a.shape[0]
    weights = np.ones(n) / n
    M = ot.dist(a, b, metric='sqeuclidean')      # squared-euclidean cost
    return ot.emd2(weights, weights, M)          # returns 2-Wasserstein squared


def run_wasserstein_abc(y_obs, budget, keep_fraction=0.02,
                        noise_sd=lv.OBS_NOISE_SD, rng=None,
                        return_info=False, verbose=True):
    """
    ABC-rejection replacing summary distance with 2-Wasserstein distance
    between simulated and observed noisy trajectories. Bernton et al. 2019.
    Each trajectory is treated as a uniform empirical measure on its 151
    (predator, prey) points; time-ordering is dropped, but the joint
    distribution of the phase-space visits is preserved.

    See run_abc_rejection for the diagnostics interpretation.
    """
    rng = np.random.default_rng() if rng is None else rng
    scale = y_obs.std(axis=0)
    scale = np.where(scale < 1e-8, 1e-8, scale)

    thetas, dists = [], []
    for _ in tqdm(range(budget), desc="Wasserstein-ABC", leave=False):
        lt = lv.sample_log_prior(rng=rng)
        traj = _safe_trajectory(lt, noise_sd, rng=rng)
        if traj is None:
            continue
        d = _wasserstein_2d(traj, y_obs, scale)
        if np.isfinite(d):
            thetas.append(lt)
            dists.append(d)

    thetas = np.array(thetas)
    dists = np.array(dists)
    n_keep = max(1, int(np.ceil(keep_fraction * len(thetas))))
    keep_idx = np.argpartition(dists, n_keep)[:n_keep]
    samples = thetas[keep_idx]

    info = dict(
        acceptance_rate = keep_fraction,
        n_valid_sims    = len(thetas),
        n_kept          = n_keep,
        eps_effective   = float(dists[keep_idx].max()),
        dist_quantiles  = dict(zip([10, 50, 90],
                                   np.percentile(dists, [10, 50, 90]).round(3))),
    )
    if verbose:
        print(f"[W-ABC]    kept {n_keep}/{len(thetas)} "
              f"(acc={keep_fraction:.1%}), "
              f"eps_effective={info['eps_effective']:.3f}, "
              f"dist_quantiles(10/50/90)={info['dist_quantiles']}")
    return (samples, info) if return_info else samples


# 4. BSL (random-walk Metropolis + Gaussian synthetic likelihood)
def _synth_loglik(log_theta, x_o, n_batch, noise_sd, shrink=0.1,
                  max_attempts_factor=3, rng=None):
    """
    Standardized + shrunk Gaussian synthetic log-likelihood.
    Standardization avoids the near-singular covariance that arises because
    the 9 summary stats span ~4 orders of magnitude in scale; shrinkage
    keeps it well-conditioned regardless of n_batch.
    """
    ss, attempts = [], 0
    max_attempts = n_batch * max_attempts_factor
    while len(ss) < n_batch and attempts < max_attempts:
        attempts += 1
        s = _safe_summary(log_theta, noise_sd, rng=rng)
        if s is not None and np.all(np.isfinite(s)):
            ss.append(s)
    if len(ss) < n_batch:
        return -np.inf, attempts

    ss = np.array(ss)
    sd = ss.std(axis=0)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    ss_std = ss / sd
    x_o_std = x_o / sd
    mu_std = ss_std.mean(axis=0)
    cov_std = (1 - shrink) * np.cov(ss_std, rowvar=False) \
              + shrink * np.eye(len(sd))
    ll_std = stats.multivariate_normal.logpdf(x_o_std, mean=mu_std,
                                              cov=cov_std, allow_singular=True)
    return ll_std - np.sum(np.log(sd)), attempts     # Jacobian


def run_bsl(x_o, budget, n_batch=50, proposal_sd=0.15, init=None,
            shrink=0.1, noise_sd=lv.OBS_NOISE_SD, rng=None,
            return_info=False, verbose=True):
    """
    Bayesian Synthetic Likelihood with isotropic Gaussian RW proposal.

    Budget accounting:
        M = budget // n_batch  MCMC steps, each doing n_batch simulator calls.

    Standardization + covariance shrinkage inside the synthetic likelihood
    (see _synth_loglik) is what makes n_batch=50 workable; without shrinkage
    the covariance is near-singular and the loglik is unusable.
    """
    rng = np.random.default_rng() if rng is None else rng
    M = budget // n_batch

    log_curr = lv.LOG_TRUE_PARAMS.copy() if init is None else np.asarray(init)
    ll_curr, sims = _synth_loglik(log_curr, x_o, n_batch, noise_sd,
                                  shrink=shrink, rng=rng)

    chain = np.zeros((M, lv.N_PARAMS))
    cov_rw = np.eye(lv.N_PARAMS) * proposal_sd ** 2
    n_accept = 0
    for t in tqdm(range(M), desc="BSL", leave=False):
        log_prop = rng.multivariate_normal(log_curr, cov_rw)
        if lv.log_prior(log_prop) == -np.inf:
            chain[t] = log_curr
            continue
        ll_prop, _ = _synth_loglik(log_prop, x_o, n_batch, noise_sd,
                                   shrink=shrink, rng=rng)
        if np.log(rng.uniform()) < (ll_prop - ll_curr):
            log_curr, ll_curr = log_prop, ll_prop
            n_accept += 1
        chain[t] = log_curr

    burn = min(M // 5, 200)
    samples = chain[burn:]

    info = dict(
        acceptance_rate = n_accept / M,
        n_iters         = M,
        n_batch         = n_batch,
        burn_in         = burn,
    )
    if verbose:
        print(f"[BSL]      acc={info['acceptance_rate']:.1%}, "
              f"M={M} steps × n_batch={n_batch}, kept={len(samples)}")
    return (samples, info) if return_info else samples


# 5. NPE  
def run_npe(x_o, budget, n_posterior_samples=4000, noise_sd=lv.OBS_NOISE_SD):
    """
    Single-round NPE (SNPE-C / APT by default in sbi). Trains a conditional
    density estimator for p(log theta | x) on `budget` prior simulations,
    then samples the posterior at x = x_o.
    """
    from sbi.inference import NPE

    prior = lv.make_torch_prior()
    theta = prior.sample((budget,))
    x = lv.lv_simulator(theta, noise_sd=noise_sd, show_progress=True)

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x, exclude_invalid_x=True).train()
    posterior = inference.build_posterior()

    x_o_torch = torch.as_tensor(x_o, dtype=torch.float32)
    return posterior.sample((n_posterior_samples,), x=x_o_torch).numpy()


# 6. NLE  
def run_nle(x_o, budget, n_posterior_samples=2000, noise_sd=lv.OBS_NOISE_SD):
    """
    Single-round NLE with a MAF likelihood surrogate. Posterior samples are
    obtained by slice-MCMC on the learned likelihood + prior.
    """
    from sbi.inference import NLE

    prior = lv.make_torch_prior()
    theta = prior.sample((budget,))
    x = lv.lv_simulator(theta, noise_sd=noise_sd, show_progress=True)

    inference = NLE(prior=prior, density_estimator='maf')
    estimator = inference.append_simulations(theta, x,
                                             exclude_invalid_x=True).train()
    posterior = inference.build_posterior(
        density_estimator=estimator, prior=prior,
        sample_with='mcmc', mcmc_method='slice_np_vectorized',
    )

    x_o_torch = torch.as_tensor(x_o, dtype=torch.float32)
    return posterior.sample((n_posterior_samples,), x=x_o_torch).numpy()
