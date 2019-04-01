import numpy as np
from sacred import Experiment
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from scipy import optimize
import scipy.stats as stats

ex = Experiment("MLE-dist-simulation")

@ex.config
def config():
    A = 1
    N = 1      # No of data points
    n = 100     # No of experiments to repeat over
    a0 = 0.5
    gamma = 0.52999894

@ex.capture
def plot_stats(distribution, estimator,range_ll, A, N, n):
    estimator_ll = []
    for i in tqdm(range(n)):
        data = distribution(size = N)
        estimator_ll.append(estimator(data))

    estimator_ll = np.array(estimator_ll)
    estimator_mean = np.mean(estimator_ll)
    estimator_var = np.var(estimator_ll)
    fit = stats.norm.pdf(data, estimator_mean, estimator_var)
    
    plt.plot(data, fit, '-o')
    plt.hist(estimator_ll, range=range_ll, bins = 100)
    
    plt.show()

    return estimator_mean, estimator_var

@ex.capture
def cauchy_likelihood(a, x, gamma):
    '''
    Function that defines the derivative likelihood for cauchy distribution
    '''
    val = 0
    for i in x:
        val = val + (i-a)/(1 + ((i-a)/gamma)**2)
    return val

@ex.capture
def cauchy_dist(size, A, gamma):
    '''
    Function to sample from the cauchy distribution
    '''
    return A + np.random.standard_cauchy(size=(10000))*gamma

@ex.capture
def find_root(x, a0, gamma, f = cauchy_likelihood):
    '''
    Using Newton-Raphson to find to root
    '''
    res = optimize.newton(f, a0, args=(x, gamma))
    return res


@ex.automain
def main(_run):
    distribution = cauchy_dist
    mean, var = plot_stats(distribution, find_root, range_ll=[0,2])
    print(mean, var)