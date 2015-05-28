__author__ = 'hahdawg'
import changepoint.bayes as bayes
import changepoint.bootstrap as btstrp
import os
import numpy as np


def load_data():
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, "wells.csv")) as f:
        lines = f.readlines()
    return np.array([float(f.strip()) for f in lines])


def test_bayes():
    xs = load_data()
    mu_hat, var_hat = bayes.find_all_cps(xs, cp_prob=1./250, plot=True)
    bayes.plot_results(xs, mu_hat, var_hat)