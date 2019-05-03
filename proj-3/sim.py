import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
from functools import partial
from sacred import Experiment
from scipy import optimize
from tqdm import tqdm

ex = Experiment("MLE-dist-simulation")


@ex.config
def config():
    # Vary
    n_ll = [10, 1000, 10000]
    m_ll = [1, 10]
    theta_0_ll = [[0.5, 0.5], [0.9, 0.2], [0.2, 0.9], [0.7, 0.4]]
    theta_0_ll = np.array(theta_0_ll)

    # Constant
    p = 0.72
    q = 0.43
    pi = 0.5

    supress = False


@ex.capture
def generate_samples(n, m, p, q, pi):
    """
    Generate n samples of length m
    using probabilities a -> p ,and b -> q
    """
    z = np.random.choice([0, 1], size=n, replace=True, p=[pi, 1 - pi])
    x = np.zeros(shape=[n, m], dtype='int')

    p_q = [p, q]
    for i, z_i in enumerate(z):
        p_i = p_q[z_i]
        x[i, :] = np.random.choice([0, 1], size=m, replace=True, p=[1 - p_i, p_i])

    return (x, z)


@ex.capture
def k_z_x(x, p_i, q_i, n, m, pi):
    """
    Computes CPD of z_i=a given x at theta(t)
    """
    h = np.sum(x, axis=1)
    joint_a = pi * np.exp(h * np.log(p_i) + (m - h) * np.log(1 - p_i))
    joint_b = (1 - pi) * np.exp(h * np.log(q_i) + (m - h) * np.log(1 - q_i))
    cpd_z = joint_a / (joint_a + joint_b)
    return cpd_z


@ex.capture
def em_step(x, theta_old, n, m, pi):
    """
    Performs one step of EM

    Returns theta_new
    """
    p_i, q_i = theta_old[0], theta_old[1]
    cpd_z = k_z_x(x, p_i, q_i, n, m)
    p_new = np.sum(cpd_z * np.sum(x, axis=1)) / (np.sum(cpd_z) * m)
    q_new = np.sum((1 - cpd_z) * np.sum(x, axis=1)) / (np.sum(1 - cpd_z) * m)
    return np.array([p_new, q_new])


@ex.automain
def main(n_ll, m_ll, theta_0_ll, p, q, pi, supress):
    pbar = tqdm(range(len(n_ll) * len(m_ll) * len(theta_0_ll)))

    if not os.path.exists("logs"):
        os.mkdir("logs")

    for n in n_ll:
        for m in m_ll:
            p_hist_EM_ll = []
            q_hist_EM_ll = []
            p_hist_ML_ll = []
            q_hist_ML_ll = []

            x, z = generate_samples(n, m)
            x_A = np.mean(x[np.where(z == 0)])
            x_B = np.mean(x[np.where(z == 1)])
            theta_ML = np.array([x_A, x_B])

            for theta_0 in theta_0_ll:
                theta_old = theta_0

                p_hist_EM = []
                q_hist_EM = []

                while True:
                    theta_new = em_step(x, theta_old, n, m)
                    if np.linalg.norm(theta_new - theta_old) <= 1e-6:
                        # print("The estimiated values are: ")
                        # print(f"p_em = {theta_new[0]}")
                        # print(f"q_em = {theta_new[1]}")
                        break
                    else:
                        p_hist_EM.append(theta_old[0])
                        q_hist_EM.append(theta_old[1])
                        theta_old = theta_new

                pbar.set_description(f"n {n} m {m} theta_0 {theta_0} theta_EM {theta_new} theta_ML {theta_ML}")

                p_hist_EM_ll.append(p_hist_EM)
                q_hist_EM_ll.append(q_hist_EM)

            f, (ax1, ax2) = plt.subplots(2, sharey = True)
            legend_p = []
            legend_q = []
            max_length = np.max([len(ll) for ll in p_hist_EM_ll])

            for e in range(len(p_hist_EM_ll)):
                nn = np.arange(len(p_hist_EM_ll[e])) + 1
                legend_p.append(f'p-EM-{theta_0_ll[e][0]}')
                legend_q.append(f'q-EM-{theta_0_ll[e][1]}')
                ax1.plot(nn, p_hist_EM_ll[e], alpha=1)
                ax2.plot(nn, q_hist_EM_ll[e], alpha=1)

            legend_p.append('p-ML')
            legend_q.append('q-ML')
            ax1.plot(np.arange(max_length)+1, [theta_ML[0]] * max_length, alpha=1)
            ax2.plot(np.arange(max_length)+1, [theta_ML[1]] * max_length, alpha=1)

            ax1.legend(legend_p)
            ax2.legend(legend_q)

            plt.savefig(f"logs/Convergence-n-{n}-m-{m}")
            plt.xlabel("Iterations n")
            ax1.set_title(f"Convergence plot n={n} m={m}")
            if supress:
                plt.close()
            else:
                plt.show()
