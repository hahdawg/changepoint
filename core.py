"""
Find changepoints using binary segmentation.
"""

import numpy as np
import bottleneck as bn

MINNOBS = 10


def welch(xs, ys):
    """
    Welch's statistic for equal means
    http://en.wikipedia.org/wiki/Welch%27s_t_test

    Parameters
    ----------
    xs: np.array
    ys: np.array

    Returns
    -------
    float
    """
    xbar, ybar = bn.nanmean(xs), bn.nanmean(ys)
    sx2, sy2 = bn.nanvar(xs) + np.spacing(1), bn.nanvar(ys) + np.spacing(1)
    return np.abs(xbar - ybar)/np.sqrt(sx2/len(xs) + sy2/len(ys))


def compute_endpoints(N, minnobs):
    return minnobs, N - minnobs


def _most_likely_cp(xs, minnobs):
    start, end = compute_endpoints(len(xs), minnobs)
    wstats = np.array([welch(xs[:i], xs[i:]) for i in xrange(start, end)])
    cp = bn.nanargmax(wstats)
    stat = wstats[cp]
    return cp + start, stat


def most_likely_cp(xs, minnobs, nsamples):
    """
    Finds the most likely changepoint in xs and the corresponding p-value.

    Parameters
    ----------
    xs: np.array
    minnobs: int
        Shortest interval to search for a changepoint.
    nsamples: int
    """
    cp, stat = _most_likely_cp(xs, minnobs)
    prob = pval(stat, xs, minnobs=minnobs, nsamples=nsamples)
    return cp, prob


def bootstrap_cps(xs, nsamples, minnobs):
    """
    Computes an array of Welch stats using bootstrapped samples from xs.
    """
    N = len(xs)
    res = np.zeros(nsamples)
    for i in xrange(nsamples):
        ys = np.random.choice(xs, N)
        _, stat = _most_likely_cp(ys, minnobs)
        res[i] = stat
    return res


def pval(stat, xs, minnobs, nsamples):
    """
    Computes the bootstrapped p-value for a Welch statistic on the sample xs.
    """
    sample = bootstrap_cps(xs, minnobs=minnobs, nsamples=nsamples)
    return 1 - bn.nanmean(sample < stat)


def _find_all_cps(xs, nsamples, index, minnobs):
    res = list()
    if xs is None or (len(xs) < (2*MINNOBS + 1)):
        return res

    cp, prob = most_likely_cp(xs, minnobs=minnobs, nsamples=nsamples)
    res.append((cp + index, prob))

    left, right = _find_all_cps(xs[:cp], minnobs=minnobs, nsamples=nsamples, index=index), \
        _find_all_cps(xs[(cp + 1):], minnobs=minnobs, nsamples=nsamples, index=cp + 1)
    if left:
        res.extend(left)
    if right:
        res.extend(right)
    return res


def find_all_cps(xs, minnobs=10, nsamples=50, crit_val=0.1):
    """
    Finds all changepoints for the sample xs where the p-value is below crit_val.

    Parameters
    ----------
    xs: np.array
    minnobs: int
        Shortest interval to search for a changepoint.
    nsamples: int
        Number of bootstrap samples for computing p-values.
    crit_val: float
        Threshold for which we'll keep changepoints.

    Returns
    -------
    [(changepoint index, pvalue)]

    Example
    -------
    >>> xs = np.zeros(300)
    >>> xs[100:] += 1.0
    >>> xs[200:] += 1.0
    >>> cps = [c for c, _ in find_all_cps(xs, crit_val=0.01)]
    >>> assert 100 in cps 
    >>> assert 200 in cps
    """
    res = [(c, p) for c, p in _find_all_cps(xs, minnobs=minnobs, index=0, nsamples=nsamples) if p < crit_val]
    return sorted(res, key=lambda t: t[0])
