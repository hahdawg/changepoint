import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt


def nct(x, mu, sigma2, dof):
    """
    Noncentral t distribution

    see http://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    c = np.exp(gammaln(0.5*dof + 0.5) - gammaln(0.5*dof))*(dof*np.pi*sigma2)**(-0.5)
    return c*(1 + (1.0/(dof*sigma2))*(x - mu)**2)**(-0.5*(dof + 1))


def update_params(x, prior, posterior):
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
    """
    rp = rs*pred_prob
    rph = rp*h
    rt = np.r_[rph.sum(), rp - rph]
    return rt/rt.sum()


def compute_t_params(mu, kappa, alpha, beta):
    """
    Transform parameters from NIG to parameters for the t-distribution.
    """
    mu_, sigma2_, dof_ = mu, beta*(kappa + 1)/(alpha*kappa), 2*alpha
    return mu_, sigma2_, dof_


def compute_t_var(sigma2, dof):
    return sigma2*dof/(dof - 2)


def find_all_cps(xs, cp_prob=1/.250, plot=False):

    T = len(xs)
    prior_params = mu0, kappa0, alpha0, beta0 = xs[0], 1., 1.01, 1.
    post_params = mu_t, kappa_t, alpha_t, beta_t = map(lambda f: np.array([f]), prior_params)

    # run length distribution
    RLD = np.zeros((T, T))
    RLD[0, 0] = 1

    # posterior mean
    M = np.zeros((T, T))
    M[0, 0] = mu0

    # posterior variance
    V = np.zeros((T, T))
    V[0, 0] = 1.0

    mu_pred, sigma2_pred, dof_pred = compute_t_params(mu_t, kappa_t, alpha_t, beta_t)
    for t, x in enumerate(xs[1:], start=1):
        pred_prob = np.array([nct(x, m, v, d) for m, v, d in zip(mu_pred, sigma2_pred, dof_pred)])

        RLD[:t + 1, t] = compute_rt(RLD[:t, t - 1], pred_prob, cp_prob)

        post_params = mu_t, kappa_t, alpha_t, beta_t = update_params(x, prior_params, post_params)
        mu_pred, sigma2_pred, dof_pred = compute_t_params(mu_t, kappa_t, alpha_t, beta_t)

        M[:t + 1, t] = mu_pred
        V[:t + 1, t] = compute_t_var(sigma2_pred, dof_pred)

    mu_hat = np.sum(M*RLD, axis=0)
    var_hat = np.sum(V*RLD, axis=0)
    if plot:
        plot_results(xs, mu_hat, var_hat)

    return mu_hat, var_hat


def plot_results(xs, mu_hat, var_hat, crit=1.645):
    axis = np.arange(len(xs))
    sd_hat = np.sqrt(var_hat)

    plt.plot(axis, xs)
    plt.plot(axis, mu_hat)
    plt.plot(axis, mu_hat - crit*sd_hat, c="grey", alpha=0.75)
    plt.plot(axis, mu_hat + crit*sd_hat, c="grey", alpha=0.75)

    plt.legend(["actual", "mean", "+ error", "- error"])
