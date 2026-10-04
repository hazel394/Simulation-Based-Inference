"""
SBI methods for posterior inference on the SNF / CER network model.

Every method has the same signature returning
    (gamma_samples, Gm_samples, acceptance_rate)

so they can be dropped into the comparison notebook interchangeably.

Simulator defaults to CER for speed; pass a custom callable
    simulator(Gm, gamma) -> (n_pop, e) population
to use SNF instead.
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm

import networks


# ============================================================
# Summary statistics
# ============================================================

def weighted_average(population):
    return np.mean(population, axis=0)

def population_variance(population):
    return np.var(population, axis=0)

def mode_network(population):
    """Threshold the population edge-average at 0.5."""
    return (weighted_average(population) >= 0.5).astype(int)

def vector_to_matrix(v, N):
    A = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    A[iu] = v
    A = A + A.T
    return A

def summ_stat_scalar(population):
    """Mean edge density + mean edge variance."""
    a = weighted_average(population)
    v = population_variance(population)
    return np.array([np.mean(a), np.mean(v)])

def summ_stat_eig(population, N):
    """Eigenvalues of the mean and variance adjacency matrices."""
    a = weighted_average(population)
    v = population_variance(population)
    eigvals_mean = np.linalg.eigvalsh(vector_to_matrix(a, N))
    eigvals_var  = np.linalg.eigvalsh(vector_to_matrix(v, N))
    return np.concatenate([eigvals_mean, eigvals_var])

def summ_stat_degree(population, N):
    """Node-wise degree sequence from mean + variance edge weights."""
    avg_edges = weighted_average(population)
    var_edges = population_variance(population)
    edge_list = list(combinations(range(N), 2))

    node_deg = np.zeros(N)
    node_var_deg = np.zeros(N)
    for idx, (u, v) in enumerate(edge_list):
        node_deg[u] += avg_edges[idx]
        node_deg[v] += avg_edges[idx]
        node_var_deg[u] += var_edges[idx]
        node_var_deg[v] += var_edges[idx]

    return np.concatenate([node_deg, node_var_deg])

def summ_stat_eig_deg(population, N):
    return np.concatenate([summ_stat_eig(population, N),
                           summ_stat_degree(population, N)])


# ============================================================
# Prior
# ============================================================

def sample_prior_Gm(N):
    return np.random.binomial(1, 0.5, size=networks.n_edges(N))

def sample_prior_gamma(low=0.0, high=3.0):
    return np.random.uniform(low, high)


# ============================================================
# Default simulator wrapper (CER for speed)
# ============================================================

def _default_simulator(n_pop):
    def sim(Gm, gamma):
        return networks.generate_population_CER(n_pop, Gm, gamma)
    return sim


# ============================================================
# Calibration for summary-statistic normalisation
# ============================================================

def calibrate_normalization(summ_stat_fn, n_pilot, N, simulator,
                            gamma_low=0.0, gamma_high=3.0):
    """
    Robust normalisation via median / IQR on prior-predictive draws.
    """
    pilot_stats = []
    for _ in range(n_pilot):
        Gm_s = sample_prior_Gm(N)
        g_s = sample_prior_gamma(gamma_low, gamma_high)
        pilot_stats.append(summ_stat_fn(simulator(Gm_s, g_s)))
    pilot_stats = np.asarray(pilot_stats)
    med = np.median(pilot_stats, axis=0)
    iqr = np.percentile(pilot_stats, 75, axis=0) - np.percentile(pilot_stats, 25, axis=0)
    iqr[iqr == 0] = 1.0
    return med, iqr


# ============================================================
# ABC with a generic (scalarised) summary statistic
# ============================================================

def abc_summary(observation, summ_stat_fn, N, n_pop, num_simulations, epsilon,
                simulator=None, n_pilot=1000,
                gamma_low=0.0, gamma_high=3.0, verbose=True):
    """
    Rejection ABC with normalised Euclidean distance on a summary statistic.
    """
    if simulator is None:
        simulator = _default_simulator(n_pop)

    med, iqr = calibrate_normalization(summ_stat_fn, n_pilot, N, simulator,
                                       gamma_low, gamma_high)
    s_y = (summ_stat_fn(observation) - med) / iqr

    Gm_acc, g_acc = [], []
    it = tqdm(range(num_simulations), desc='ABC') if verbose else range(num_simulations)
    for _ in it:
        Gm_s = sample_prior_Gm(N)
        g_s = sample_prior_gamma(gamma_low, gamma_high)
        s_x = (summ_stat_fn(simulator(Gm_s, g_s)) - med) / iqr
        if np.linalg.norm(s_y - s_x) < epsilon:
            Gm_acc.append(Gm_s)
            g_acc.append(g_s)

    return (np.asarray(g_acc),
            np.asarray(Gm_acc),
            len(g_acc) / num_simulations)


# ============================================================
# ABC with Wasserstein-Hamming distance between populations
# ============================================================

def abc_wasserstein(observation, N, n_pop, num_simulations, epsilon,
                    simulator=None,
                    gamma_low=0.0, gamma_high=3.0, verbose=True):
    """
    Rejection ABC with Wasserstein distance on the population empirical
    measure, using Hamming as ground cost between individual networks.
    """
    import ot
    if simulator is None:
        simulator = _default_simulator(n_pop)

    e = networks.n_edges(N)
    a = np.ones(len(observation)) / len(observation)

    def w_dist(pop_obs, pop_sim):
        cost = ot.dist(pop_obs, pop_sim, metric='hamming') * pop_obs.shape[1]
        b = np.ones(len(pop_sim)) / len(pop_sim)
        return ot.emd2(a, b, cost)

    Gm_acc, g_acc = [], []
    it = tqdm(range(num_simulations), desc='ABC-Wass') if verbose else range(num_simulations)
    for _ in it:
        Gm_s = sample_prior_Gm(N)
        g_s = sample_prior_gamma(gamma_low, gamma_high)
        d = w_dist(observation, simulator(Gm_s, g_s)) / e
        if d < epsilon:
            Gm_acc.append(Gm_s)
            g_acc.append(g_s)

    return (np.asarray(g_acc),
            np.asarray(Gm_acc),
            len(g_acc) / num_simulations)


# ============================================================
# ABC with Hamming distance between mode networks
# ============================================================

def abc_hamming_mode(observation, N, n_pop, num_simulations, epsilon,
                     simulator=None,
                     gamma_low=0.0, gamma_high=3.0, verbose=True):
    """
    Rejection ABC using Hamming distance between the observed and
    simulated mode networks (both thresholded at 0.5).
    """
    if simulator is None:
        simulator = _default_simulator(n_pop)

    e = networks.n_edges(N)
    s_y = mode_network(observation)

    Gm_acc, g_acc = [], []
    it = tqdm(range(num_simulations), desc='ABC-Hamm') if verbose else range(num_simulations)
    for _ in it:
        Gm_s = sample_prior_Gm(N)
        g_s = sample_prior_gamma(gamma_low, gamma_high)
        d = networks.hamming(s_y, mode_network(simulator(Gm_s, g_s))) / e
        if d < epsilon:
            Gm_acc.append(Gm_s)
            g_acc.append(g_s)

    return (np.asarray(g_acc),
            np.asarray(Gm_acc),
            len(g_acc) / num_simulations)


# ============================================================
# Exact CER posterior (closed-form up to a 1D integral)
# ============================================================

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

def cer_true_posterior_samples(observation, gamma_low=0.0, gamma_high=3.0,
                               n_grid=2000, n_samples=2000, seed=None):
    """
    Exact samples from the CER joint posterior p(gamma, Gm | data) under
    Uniform(gamma_low, gamma_high) x prod_i Bern(0.5) priors.

    The CER likelihood factorises per edge:
        L(gamma, Gm | data) = prod_i (1-alpha)^{m_i} * alpha^{f_i}
    where alpha = 1/(1+exp(gamma)), m_i is matches and f_i is flips for edge i.
    Marginalising Gm_i in closed form gives a 1D marginal p(gamma | data);
    the conditional p(Gm_i | gamma, data) is Bernoulli.

    Same return signature as the other methods.
    """
    rng = np.random.default_rng(seed)
    n, e = observation.shape
    s = observation.sum(axis=0)                             # (e,)

    # marginal log-likelihood of gamma on a grid (log-space for stability)
    gamma_grid = np.linspace(gamma_low, gamma_high, n_grid)
    log_ll = np.empty(n_grid)
    for gi, gm in enumerate(gamma_grid):
        alpha = 1.0 / (1.0 + np.exp(gm))
        la, l1a = np.log(alpha), np.log1p(-alpha)
        log_L1 = s * l1a + (n - s) * la
        log_L0 = s * la  + (n - s) * l1a
        log_ll[gi] = np.logaddexp(log_L0, log_L1).sum()

    pdf = np.exp(log_ll - log_ll.max())
    pdf /= _trapz(pdf, gamma_grid)
    dg = gamma_grid[1] - gamma_grid[0]
    w = pdf * dg; w /= w.sum()
    gamma_samples = rng.choice(gamma_grid, size=n_samples, p=w)

    # conditional Gm | gamma, data (per edge Bernoulli)
    Gm_samples = np.empty((n_samples, e), dtype=int)
    for k, gm in enumerate(gamma_samples):
        alpha = 1.0 / (1.0 + np.exp(gm))
        la, l1a = np.log(alpha), np.log1p(-alpha)
        log_L1 = s * l1a + (n - s) * la
        log_L0 = s * la  + (n - s) * l1a
        p1 = np.exp(log_L1 - np.logaddexp(log_L0, log_L1))
        Gm_samples[k] = rng.binomial(1, p1)

    return gamma_samples, Gm_samples, 1.0


# ============================================================
# MNPE (mixed neural posterior estimation)
# ============================================================

def run_mnpe(observation, N, n_pop, num_simulations, num_posterior_samples=2000,
             gamma_low=0.0, gamma_high=3.0, simulator=None, verbose=True):
    """
    Mixed Neural Posterior Estimation on (gamma, Gm) with the population
    mean edge vector as summary.
    """
    import torch
    from sbi.inference import MNPE
    from sbi.utils import BoxUniform

    if simulator is None:
        simulator = _default_simulator(n_pop)

    d = networks.n_edges(N)
    low  = torch.cat([torch.tensor([gamma_low]),  torch.zeros(d)])
    high = torch.cat([torch.tensor([gamma_high]), torch.ones(d)])
    prior = BoxUniform(low=low, high=high)

    theta_list, x_list = [], []
    it = tqdm(range(num_simulations), desc='MNPE-sim') if verbose else range(num_simulations)
    for _ in it:
        Gm_s = sample_prior_Gm(N)
        g_s = sample_prior_gamma(gamma_low, gamma_high)
        theta_list.append(np.concatenate([[g_s], Gm_s]))
        x_list.append(simulator(Gm_s, g_s).mean(axis=0))

    theta = torch.tensor(np.array(theta_list), dtype=torch.float32)
    x     = torch.tensor(np.array(x_list),     dtype=torch.float32)

    inference = MNPE(prior=prior)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    x_obs = observation.mean(axis=0)
    samples = posterior.sample(
        (num_posterior_samples,),
        x=torch.tensor(x_obs, dtype=torch.float32),
        reject_outside_prior=False,
    ).detach().numpy()

    return samples[:, 0], samples[:, 1:].astype(int), 1.0
