"""
Uses Adams and MacKay's algorithm from 

    https://hips.seas.harvard.edu/files/adams-changepoint-tr-2007.pdf

to find changepoints for Gaussian time series.
"""
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt


def nct(x, mu, sigma2, dof):
    """
    Noncentral t density.

    see http://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    c = np.exp(gammaln(0.5*dof + 0.5) - gammaln(0.5*dof))*(dof*np.pi*sigma2)**(-0.5)
    return c*(1 + (1.0/(dof*sigma2))*(x - mu)**2)**(-0.5*(dof + 1))


def update_params(x, prior, posterior):
    """
    Updates posterior parameters, given a new datum x.
    """
    mu0, kappa0, alpha0, beta0 = prior
    mu_t, kappa_t, alpha_t, beta_t = posterior
    return np.r_[mu0, (kappa_t*mu_t + x)/(kappa_t + 1)], \
        np.r_[kappa0, kappa_t + 1], \
        np.r_[alpha0, alpha_t + 0.5], \
        np.r_[beta0, beta_t + 0.5*kappa_t*(x - mu_t)**2/(kappa_t + 1)]


def compute_rt(rs, pred_prob, h):
    """
    Computes all run probabilities for time t, using run probabilities for time s,
    predictive probabilities, and prior probability of a change point.

    At time s, rs will be like [p0, p1, ..., ps], where pi is the probability that 
    the run length at time s was i.  The probability of a change point is the sum of 
    the probabilities of each run length being zero.  For each pi in rs, the probability
    that the run length increases by 1 is represented by the array of growth probabilities.
    """
    rp = rs*pred_prob
    rph = rp*h
    cp_prob = rph.sum()
    growth_prob = rp - rph
    rt = np.r_[cp_prob, growth_prob]
    return rt/rt.sum()


def compute_t_params(mu, kappa, alpha, beta):
    """
    Transform parameters from NIG to parameters for the t-distribution.
    """
    mu_, sigma2_, dof_ = mu, beta*(kappa + 1)/(alpha*kappa), 2*alpha
    return mu_, sigma2_, dof_


def compute_t_var(sigma2, dof):
    return sigma2*dof/(dof - 2)


def find_all_cps(xs, cp_prob=1./250, plot=False):
    """
    Find changepoints for a Gaussian time series xs.

    Parameters
    ----------
    xs: np.array
    cp_prob: float
        Prior probability of changepoint.
    plot: bool

    Returns
    -------
    (R, M, V)
        R: run length probabilities for (run length, time index)
        M: posterior means for (run length, time index)
        V: posterior variances for (run length, time index)
    Rows are run lengths, columns are time indexes.  All matrices are upper triangular, because 
    the run length can't be greater than the index of the current period.

    Example
    -------
    >>> xs = np.zeros(100)
    >>> xs[50] += 5.0
    >>> R, M, V = find_all_cps(xs)
    >>> mu_hat = np.sum(R*M, axis=0)
    >>> assert mu_hat[49] < mu_hat[50]
    """
    prior_params = mu0, kappa0, alpha0, beta0 = np.mean(xs), 1., 1.01, 1.
    post_params = mu_t, kappa_t, alpha_t, beta_t = map(lambda f: np.array([f]), prior_params)

    T = len(xs)
    R, M, V = np.zeros((T, T)), np.zeros((T, T)), np.zeros((T, T))
    R[0, 0] = 1
    M[0, 0] = mu0
    V[0, 0] = xs.var()

    mu_pred, sigma2_pred, dof_pred = compute_t_params(mu_t, kappa_t, alpha_t, beta_t)
    for t, x in enumerate(xs[1:], start=1):
        pred_prob = np.array([nct(x, m, v, d) for m, v, d in zip(mu_pred, sigma2_pred, dof_pred)])

        R[:t + 1, t] = compute_rt(R[:t, t - 1], pred_prob, cp_prob)

        post_params = mu_t, kappa_t, alpha_t, beta_t = update_params(x, prior_params, post_params)
        mu_pred, sigma2_pred, dof_pred = compute_t_params(mu_t, kappa_t, alpha_t, beta_t)

        M[:t + 1, t] = mu_pred
        V[:t + 1, t] = compute_t_var(sigma2_pred, dof_pred)

    if plot:
        mu_hat = np.sum(M*R, axis=0)
        var_hat = np.sum(V*R, axis=0)
        plot_results(xs, mu_hat, var_hat)

    return R, M, V 


def plot_results(xs, mu_hat, var_hat, crit=1.645):
    axis = np.arange(len(xs))
    sd_hat = np.sqrt(var_hat)

    plt.plot(axis, xs)
    plt.plot(axis, mu_hat)
    plt.plot(axis, mu_hat - crit*sd_hat, c="grey", alpha=0.75)
    plt.plot(axis, mu_hat + crit*sd_hat, c="grey", alpha=0.75)

    plt.legend(["actual", "mean", "+ error", "- error"])
    plt.show()
