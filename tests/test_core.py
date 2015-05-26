import changepoint.core as cp
import numpy as np


def test_most_likely_cp():

    ws = np.r_[np.zeros(51), np.ones(50)]
    loc, _ = cp.most_likely_cp(ws, minnobs=10, nsamples=10)
    assert loc == 51

    xs = np.r_[np.zeros(50), np.ones(50)]
    loc, _ = cp.most_likely_cp(xs, minnobs=10, nsamples=10)
    assert loc == 50

    ys = np.r_[np.zeros(10), np.ones(90)]
    loc, _ = cp.most_likely_cp(ys, minnobs=10, nsamples=10)
    assert loc == 10

    zs = np.r_[np.zeros(90), np.ones(10)]
    loc, _ = cp.most_likely_cp(zs, minnobs=10, nsamples=10)
    assert loc == 89

    print "most_likely_cp tests passed"


def test_find_all_cps():
    extract_cps = lambda ys: [y[0] for y in ys]

    even = np.r_[np.zeros(20), np.ones(20), np.zeros(20), np.ones(30)]
    assert extract_cps(cp.find_all_cps(even)) == [20, 40, 60]
    
    odd = np.r_[np.zeros(20), np.ones(15), np.zeros(20), np.ones(30)]
    assert extract_cps(cp.find_all_cps(odd)) == [20, 35, 55]

    print "find_all_cps tests passed"


if __name__ == "__main__":
    test_most_likely_cp()
    test_find_all_cps()
