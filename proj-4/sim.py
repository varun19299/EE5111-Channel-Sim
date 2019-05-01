import numpy as np
from sacred import Experiment
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from scipy import optimize
import scipy.stats as stats

ex = Experiment("Bayes-Dist-Sim")

@ex.config
def config():
    n = 1000
    cov_mat = np.array([[1,0],[0,2]])
    mean_vec = np.array([0,0])
    v_0 = 5
    delta_0 = np.array([[4,0],[0,5]])
    m = 1000
    A_1 = 0.05
    A_2 = 0.05

@ex.capture
def generate_samples(n,mean_vec,cov_mat):
    """
    Generate n samples from multivariate Gaussian dist
    """
    Y = np.random.multivariate_normal(mean_vec,cov_mat,n)
    return Y

@ex.capture
def max_likelihood(Y,n):

    s = np.zeros([2,2])
    for i in range(n):
        s = s + np.outer(Y[i],Y[i])
    s = s/n
    
    print("MLE: "+str(s))

@ex.capture
def bayes_estim(Y,v_0,delta_0,n):
    
    s = np.zeros([2,2])
    for i in range(n):
        s = s + np.outer(Y[i],Y[i])
    
    v_n = v_0 + n
    delta_n = delta_0 + s

    generator = stats.invwishart(df = v_n,scale=delta_n)

    mmse = generator.mean()
    mAP = generator.mode()

    print("\nBayes MMSE: " + str(mmse))
    print("\nBayes mAP: "+ str(mAP))

@ex.capture
def bayes_monte_carlo(Y,v_0,delta_0,n,m):
    
    s = np.zeros([2,2])
    for i in range(n):
        s = s + np.outer(Y[i],Y[i])
    
    v_n = v_0 + n
    delta_n = delta_0 + s

    generator = stats.invwishart(df = v_n,scale=delta_n)
    
    numer = np.zeros([2,2])
    denom = 0
    
    for i in range(m):

        sigma = generator.rvs(size=1)

        numer = numer + (1/(i+1))*(sigma-numer)

        denom = denom + (1/(i+1))*(1-denom)
    
    print("\nMC integ: "+str(numer/denom))

@ex.capture
def gibbs_sampler(Y,v_0,A_1,A_2,n):

    s = np.zeros([2,2])
    for i in range(n):
        s = s + np.outer(Y[i],Y[i])        

    v_n = v_0 + n
    
    a_1 = stats.invgamma.rvs(scale = 0.5, a = 1/(A_1)**2,size=1)
    a_2 = stats.invgamma.rvs(scale = 0.5, a = 1/(A_2)**2,size=1)
    a_1=0.5
    a_2 = 0.5
    cov_mat = np.zeros([2,2])
    for k in range(1000):
        sigma = stats.invwishart.rvs(df=v_0+1+n,scale=2*v_0*np.diag([a_1,a_2])+s,size=1)
        cov_mat = cov_mat + (1/(k+1))*(sigma-cov_mat)
        
        a_1 = stats.invgamma.rvs(scale=(v_0+n)/2,a=v_0*np.linalg.inv(sigma)[0,0]+1/(A_1)**2,size=1)
        a_2 = stats.invgamma.rvs(scale=(v_0+n)/2,a=v_0*np.linalg.inv(sigma)[1,1]+1/(A_2)**2,size=1)

    print("\nGibbs estim: "+str(cov_mat))

@ex.automain
def main(_run):
    Y = generate_samples()

    max_likelihood(Y)
    bayes_estim(Y)
    bayes_monte_carlo(Y)
    gibbs_sampler(Y)