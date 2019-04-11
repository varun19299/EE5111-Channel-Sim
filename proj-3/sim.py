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
    n = 10000
    m = 10
    p = 0.72
    q = 0.43
    pi = 0.5
    p_0 = 0.7
    q_0 = 0.4

@ex.capture
def generate_samples(n,m,p,q,pi):
    """
    Generate n samples of length m
    using probabilities a -> p ,and b -> q
    """
    z = np.random.choice([0,1],size=n,replace=True,p=[pi,1-pi])
    x = np.zeros(shape=[n,m],dtype='int')

    p_q = [p,q]    
    for i,z_i in enumerate(z):
        p_i = p_q[z_i]
        x[i,:] = np.random.choice([0,1],size=m,replace=True,p=[1-p_i,p_i])
    
    return (x,z)

@ex.capture
def k_z_x(x,p_i,q_i,pi,n,m):
    """
    Computes CPD of z_i=a given x at theta(t)
    """
    h = np.sum(x,axis=1)
    joint_a = pi*np.exp(h*np.log(p_i)+(m-h)*np.log(1-p_i))
    joint_b = (1-pi)*np.exp(h*np.log(q_i)+(m-h)*np.log(1-q_i))
    cpd_z = joint_a/(joint_a+joint_b)
    return cpd_z

@ex.capture
def em_step(x,p_i,q_i,pi,n,m):
    """
    Performs one step of EM
    """
    cpd_z = k_z_x(x,p_i,q_i)
    p_new = np.sum(cpd_z*np.sum(x,axis=1))/(np.sum(cpd_z)*m)
    q_new = np.sum((1-cpd_z)*np.sum(x,axis=1))/(np.sum(1-cpd_z)*m)
    return p_new,q_new

@ex.automain
def main(_run,p_0,q_0):
    x,z = generate_samples()

    p_i = p_0
    q_i = q_0

    while True:
        p_i_1,q_i_1 = em_step(x,p_i,q_i)
        theta_old = np.array([p_i,q_i])
        theta_new = np.array([p_i_1,q_i_1])
        p_i = p_i_1
        q_i = q_i_1
        if np.linalg.norm(theta_old-theta_new)<=1e-6:
            print("The estimiated values are: ")
            print("p_em = "+str(p_i))
            print("q_em = "+str(q_i))
            break