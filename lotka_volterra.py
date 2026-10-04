"""
Stochastic Lotka-Volterra model (4-reaction continuous-time Markov jump process).

Canonical benchmark following Papamakarios & Murray (2016).

State:  s = [predator, prey]
Reactions:
    predator born   rate  theta_1 * x * y
    predator dies   rate  theta_2 * x
    prey born       rate  theta_3 * y
    prey eaten      rate  theta_4 * x * y

All inference is done on log-parameters (log-uniform prior over a wide box).
All data-generating pipelines add iid Gaussian observation noise so that every
method sees exactly the same observation model.
"""

import numpy as np
import torch
from tqdm.auto import tqdm


# Simulation constants
INIT = [50, 100]              # initial [predator, prey]
DT = 0.2                      # observation-grid spacing
DURATION = 30                 # total simulated time
MAX_N_STEPS = 10_000          # safety cap on Gillespie events
N_TIME_POINTS = int(DURATION / DT) + 1     # = 151


# Parameters

N_PARAMS = 4
TRUE_PARAMS = np.array([0.01, 0.5, 1.0, 0.01])
LOG_TRUE_PARAMS = np.log(TRUE_PARAMS)
PARAM_NAMES = [r"$\log\theta_1$", r"$\log\theta_2$",
               r"$\log\theta_3$", r"$\log\theta_4$"]

LOG_PRIOR_LOW = -5.0
LOG_PRIOR_HIGH = 2.0

OBS_NOISE_SD = 5.0            # iid Gaussian noise on each recorded state

N_SUMMARY = 9

_rng = np.random.default_rng(0)


def set_seed(seed):
    """Reset both numpy and torch RNGs for reproducibility."""
    global _rng
    _rng = np.random.default_rng(seed)
    torch.manual_seed(seed)


def _get_rng(rng):
    return _rng if rng is None else rng


# 1. Gillespie simulator (exact stochastic trajectory)

def simulate_lotka_volterra(params, init=INIT, dt=DT, duration=DURATION,
                            max_n_steps=MAX_N_STEPS, rng=None):
    """
    Exact Gillespie simulation of the 4-reaction stochastic LV process.
    Returns the state recorded on a regular grid of length N_TIME_POINTS.

    Parameters
    ----------
    params : (4,) natural-scale rate parameters (not log-scale)

    Returns
    -------
    states : (N_TIME_POINTS, 2) trajectory [predator, prey]

    Raises
    ------
    RuntimeError  if the number of events exceeds `max_n_steps`
                  (blown-up trajectory).
    """
    rng = _get_rng(rng)
    params = np.asarray(params, dtype=float)
    state = np.asarray(init, dtype=float).copy()

    num_rec = int(duration / dt) + 1
    states = np.zeros((num_rec, state.size))

    time = 0.0
    cur_time = 0.0
    n_steps = 0

    for i in range(num_rec):
        while cur_time > time:
            x, y = state
            xy = x * y
            rates = params * np.array([xy, x, y, xy])
            total_rate = rates.sum()

            if total_rate == 0:
                time = float('inf')          # extinction
                break

            time += rng.exponential(1.0 / total_rate)
            reaction = rng.choice(4, p=rates / total_rate)

            if reaction == 0:
                state[0] += 1                # predator born
            elif reaction == 1:
                state[0] -= 1                # predator dies
            elif reaction == 2:
                state[1] += 1                # prey born
            else:
                state[1] -= 1                # prey eaten

            n_steps += 1
            if n_steps > max_n_steps:
                raise RuntimeError(f'Simulation exceeded {max_n_steps} steps.')

        states[i] = state.copy()
        cur_time += dt

    return states


# 2. Observation model

def simulate_noisy_trajectory(params, noise_sd=OBS_NOISE_SD, init=INIT,
                              dt=DT, duration=DURATION, rng=None):
    """
    Simulate the true LV process and add iid Gaussian observation noise.
    THIS is the observation model that every inference method is targeting.
    """
    rng = _get_rng(rng)
    states = simulate_lotka_volterra(params, init=init, dt=dt,
                                     duration=duration, rng=rng)
    return states + rng.normal(0, noise_sd, size=states.shape)



# 3. Summary statistics (9-dim, standard P&M 2016 vector)

def calc_summary_stats(states):
    """
    Map a (T, 2) trajectory to a 9-dim summary vector:
        [ mean(x), mean(y),
          log(var(x)+1), log(var(y)+1),
          acf(x, lag=1), acf(x, lag=2),
          acf(y, lag=1), acf(y, lag=2),
          cross-corr(x, y) ].
    """
    N = states.shape[0]
    x, y = states[:, 0].copy(), states[:, 1].copy()

    mx, my = np.mean(x), np.mean(y)
    s2x, s2y = np.var(x, ddof=1), np.var(y, ddof=1)

    # standardize before computing correlations
    x = (x - mx) / np.sqrt(s2x)
    y = (y - my) / np.sqrt(s2y)

    acx = [np.dot(x[:-lag], x[lag:]) / (N - 1) for lag in (1, 2)]
    acy = [np.dot(y[:-lag], y[lag:]) / (N - 1) for lag in (1, 2)]
    ccxy = np.dot(x, y) / (N - 1)

    return np.array([mx, my,
                     np.log(s2x + 1), np.log(s2y + 1),
                     acx[0], acx[1],
                     acy[0], acy[1],
                     ccxy])


# 4. Prior helpers

def log_prior(log_theta):
    """Log density of the log-uniform box prior (0 inside, -inf outside)."""
    log_theta = np.asarray(log_theta)
    if np.all(log_theta >= LOG_PRIOR_LOW) and np.all(log_theta <= LOG_PRIOR_HIGH):
        return 0.0
    return -np.inf


def sample_log_prior(size=None, rng=None):
    """Draw log-theta from the prior. size=None gives a single (4,) vector."""
    rng = _get_rng(rng)
    if size is None:
        return rng.uniform(LOG_PRIOR_LOW, LOG_PRIOR_HIGH, size=N_PARAMS)
    return rng.uniform(LOG_PRIOR_LOW, LOG_PRIOR_HIGH, size=(size, N_PARAMS))


def make_torch_prior():
    """Return the sbi BoxUniform prior over log-theta (for NPE / NLE)."""
    from sbi.utils import BoxUniform
    return BoxUniform(low=LOG_PRIOR_LOW * torch.ones(N_PARAMS),
                      high=LOG_PRIOR_HIGH * torch.ones(N_PARAMS))



# 5. sbi-compatible batched simulator (log-theta -> summary stats)

def lv_simulator(log_theta, noise_sd=OBS_NOISE_SD, show_progress=False):
    """
    Vectorized simulator for sbi (NPE / NLE).
    Takes log-theta batch (n, 4), returns summary-stat tensor (n, 9).
    Blown-up simulations produce a NaN row (filtered by sbi's exclude_invalid_x).
    """
    log_theta = np.atleast_2d(np.asarray(log_theta))
    iterator = tqdm(log_theta, desc="Simulating LV") if show_progress else log_theta
    out = []
    for t in iterator:
        try:
            states = simulate_noisy_trajectory(np.exp(t), noise_sd=noise_sd)
            s = calc_summary_stats(states)
        except RuntimeError:
            s = np.full(N_SUMMARY, np.nan)
        out.append(s)
    return torch.as_tensor(np.array(out), dtype=torch.float32)


# 6. Observed dataset generator

def make_observation(true_params=TRUE_PARAMS, noise_sd=OBS_NOISE_SD, seed=1):
    """
    Generate the single observed dataset that every method conditions on.

    Returns
    -------
    y_obs : (N_TIME_POINTS, 2) noisy trajectory   (used by Wasserstein-ABC)
    x_o   : (9,) summary statistic vector          (used by every other method)
    """
    rng = np.random.default_rng(seed)
    y_obs = simulate_noisy_trajectory(true_params, noise_sd=noise_sd, rng=rng)
    x_o = calc_summary_stats(y_obs)
    return y_obs, x_o
