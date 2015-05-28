__author__ = 'hahdawg'
import changepoint.bayes as bayes
import changepoint.bootstrap as btstrp
import os
import numpy as np
import matplotlib.pyplot as plt
import time


def timed(f):

    def _f(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print end - start

    return _f


def load_data():
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, "wells.csv")) as f:
        lines = f.readlines()
    return np.array([float(f.strip()) for f in lines])[1500:2500]


@timed
def binseg_example():
    xs = load_data() 
    cps = map(lambda x:x[0], btstrp.find_all_cps(xs, crit_val=0.01))
    #plt.plot(np.arange(len(xs)), xs)
    #for cp in cps:
    #    plt.axvline(x=cp, c="grey", lw=2.0)
    #plt.show()


@timed
def bayes_example():
    xs = load_data()
    R, M, V = bayes.find_all_cps(xs, cp_prob=1./250, plot=False)
    mu_hat = np.sum(R*M, axis=0)
    var_hat = np.sum(R*V, axis=0)
    #bayes.plot_results(xs, mu_hat, var_hat)


if __name__ == "__main__":
    binseg_example()
    bayes_example()
