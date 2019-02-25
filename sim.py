import numpy as np
from sacred import Experiment
import cmath
from tqdm import tqdm
import matplotlib.pyplot as plt

ex = Experiment("simulation")

@ex.config
def config():
    x_ii = np.array([1 + 1j,-1 + 1j,1 - 1j,-1 - 1j], dtype= np.complex_)    # diag of X belongs to
    N = 1024                          # no of freqs
    L = 16                            # no of channel taps
    trials = 100                    # no of trials per estimation

    sigma = 0.1
    sigma_real = sigma/2
    sigma_imag = sigma/2

    d = 0.2                           # Decay factor
    guard = 200                         # Guard Bands
    
    lam = 0.0
    non_zero_ind = [2,5,7,9,10,12]
    zero_ind = np.delete(np.arange(L),non_zero_ind)
    eq_cons_ind = np.array([[0,2,4],[1,3,5]])


@ex.capture
def get_F(N, L):
    '''
    Generate IDFT Matrix
    : N : No of frequencies
    : n : channel taps 
    
    Returns:
    Float Matrix: N*n
    '''
    F = np.zeros((N,L), dtype = np.complex_) 
    for i in range(N):
        for j in range(L):
            F[i, j] = cmath.rect(1, 2*np.pi*i*j/1024)

    return F

@ex.capture
def get_X(xx, N, guard):
    '''
    Generates diagonal matrix with known symbols
    : Sequence xx of required length
    '''
    assert not ((N%4)&(guard%4)&(N<2*guard)), "N must be divisible by 4"

    # xx = np.tile(x_ii, (N-2*guard)//4)
    X = np.zeros([N,N], dtype=np.complex_)
    X[guard:N-guard,guard:N-guard] = np.diag(xx)
    return X

@ex.capture
def get_h(L, d=0.2, sigma2_real=0.5, sigma2_imag=0.5):
    '''
    n tap time domain channel vector

    Generates h vec according to exponentially decaying power-delay profile
    : sigma_real : variance of zero mean gaussian dist for real part
    : sigma_imag : variance of zero mean gaussian dist for imag part
    '''
    a = np.random.normal(0, np.sqrt(sigma2_real), size = L)
    b = np.random.normal(0, np.sqrt(sigma2_imag), size = L)
    p = np.exp(-d*np.arange(L))
    h = p * (a + b*1j)
    h = h/np.linalg.norm(p)
    return h[:,None]

@ex.capture
def get_h_sparse(L, zero_ind,d=0.2, sigma2_real=0.5, sigma2_imag=0.5):
    '''
    n tap time domain channel sparse vector

    Generates h vec according to exponentially decaying power-delay profile
    : sigma_real : variance of zero mean gaussian dist for real part
    : sigma_imag : variance of zero mean gaussian dist for imag part
    '''
    h = get_h(L,d,sigma2_real,sigma2_imag)
    h[zero_ind] = 0 + 0j
    return h

@ex.capture
def get_y(X,F,h,sigma_imag=0.1,sigma_real=0.05):
    '''
    Generates y vector

    Returns: 
    : y of size N 
    : h_actual of size n
    '''
    y = np.matmul(np.matmul(X,F),h)
    N = y.shape[0]
    y = y + (np.random.normal(0,np.sqrt(sigma_real),size=N) + np.random.normal(0,np.sqrt(sigma_imag),size=N) * 1j)[:,None]
    return y

@ex.capture
def get_A(X,F):
    return np.matrix(np.matmul(X,F),dtype=np.complex_)

@ex.capture
def get_estim_h(A,y,lam=0):
    '''
    LSE estimate of Ah = y 
    Regularisation lambda
    '''
    h_est = np.linalg.inv(np.matmul(A.getH(),A)+lam*np.eye(A.shape[1],dtype=np.complex_))
    h_est = np.matmul(h_est,A.getH())
    h_est = np.matmul(h_est,y)
    return h_est

@ex.capture
def get_estim_sparse_h(A,y,zero_ind,lam=0):
    '''
    LSE estimate of Ah = y 
    Regularisation lambda
    With sparsity in known indices
    '''
    h_est = get_estim_h(A,y)
    L = h_est.shape[0]
    r = len(zero_ind)
    C_m = np.eye(L)
    C_m = np.matrix(C_m[zero_ind,:],dtype=np.complex_)
    c_b = np.zeros([r,1],dtype=np.complex_)
    tempM = np.matmul(np.linalg.inv(np.matmul(A.getH(),A)+lam*np.eye(A.shape[1],dtype=np.complex_)),C_m.getH())
    constraint = np.matmul(C_m,h_est) - c_b
    h_sparse_estim = h_est - np.matmul(np.matmul(tempM,np.linalg.inv(np.matmul(C_m,tempM))),constraint)
    return h_sparse_estim

@ex.capture
def get_estim_cons_h(A,y,C_m,c_b,lam=0):
    '''
    LSE estimate of Ah = y 
    Regularisation lambda
    With known linear constraints
    '''
    h_est = get_estim_h(A,y)
    L = h_est.shape[0]
    tempM = np.matmul(np.linalg.inv(np.matmul(A.getH(),A)+lam*np.eye(A.shape[1],dtype=np.complex_)),C_m.getH())
    constraint = np.matmul(C_m,h_est) - c_b
    h_sparse_estim = h_est - np.matmul(np.matmul(tempM,np.linalg.inv(np.matmul(C_m,tempM))),constraint)
    return h_sparse_estim

@ex.capture
def get_random_xx(x_ii, N):
    '''
    Generate random sequence of symbols
    '''
    xx = np.random.randint(low=0, high=4, size=N)
    xx = xx*(1 + 0j)
    for i in range(4):
        xx[np.where(xx == i)] = x_ii[i]

    return xx

@ex.capture
def q1(N,L, trials):
    '''
    LSE of h_act for 10,000 trials
    '''
    print("\n\n -----\n Question 1\n LSE of h_act for 10,000 trials")
    h_act = get_h()
    h_est = np.zeros_like(h_act, dtype=np.complex_)
    error = 0
    pbar = tqdm(range(trials))
    for trial in pbar:
        xx = get_random_xx()
        X = get_X(xx)
        F = get_F()
        y = get_y(X,F,h_act)
        A = get_A(X,F)
        h_est_tt = get_estim_h(A,y)
        error_tt = np.linalg.norm(h_act - h_est_tt)
        h_est += (h_est_tt - h_est)/(trial + 1)
        error += (error_tt - error)/(trial + 1)
        pbar.set_description(f"Trial {trial} error {error}")
    
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {error}")
    return h_act,h_est

@ex.capture
def q2(N,L, trials):
    '''
    LSE of sparse h_act for 10,000 trials
    '''
    print("\n\n -----\n Question 2\n LSE of sparse h_act for 10,000 trials")
    h_act = get_h_sparse()
    h_est = np.zeros_like(h_act, dtype=np.complex_)
    error = 0
    pbar = tqdm(range(trials))
    for trial in pbar:
        xx = get_random_xx()
        X = get_X(xx)
        F = get_F()
        y = get_y(X,F,h_act)
        A = get_A(X,F)
        h_est_tt = get_estim_sparse_h(A,y)
        error_tt = np.linalg.norm(h_act - h_est_tt)
        h_est += (h_est_tt - h_est)/(trial + 1)
        error += (error_tt - error)/(trial + 1)
        pbar.set_description(f"Trial {trial} error {error}")
    
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {error}")
    return h_act,h_est

@ex.capture
def q3(guard, N, L, lam,trials):
    '''
    200 zero guard bands
    '''
    print("\n\n -----\n Question 3\n 200 zero guard bands, each side")
    h_act = get_h()
    error = 0
    h_est = np.zeros_like(h_act, dtype=np.complex_)
    pbar = tqdm(range(trials))
    for trial in pbar:
        xx = get_random_xx()[:N-2*guard]
        X = get_X(xx)
        F = get_F()
        y = get_y(X,F,h_act)
        A = get_A(X,F)

        h_est_tt = get_estim_h(A,y)
        error = np.linalg.norm(h_act - h_est_tt)
        error_tt = np.linalg.norm(h_act - h_est_tt)

        h_est += (h_est_tt - h_est)/(trial + 1)
        error += (error_tt - error)/(trial + 1)

        pbar.set_description(f"Trial {trial} error {error}")
    
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {error}")
    return h_act,h_est

@ex.capture
def q4(N,L,eq_cons_ind,trials):
    '''
    LSE of constrained h_act for 10,000 trials
    '''
    print("\n\n -----\n Question 4\n LSE of constrained h_act for 10,000 trials")
    h_act = get_h()
    # h_act[eq_cons_ind[1]] = h_act[eq_cons_ind[0]]
    C_m = np.zeros([eq_cons_ind.shape[1],L],dtype=np.complex_)
    C_m[np.arange(eq_cons_ind.shape[1]),eq_cons_ind[0]] = 1 + 0j
    C_m[np.arange(eq_cons_ind.shape[1]),eq_cons_ind[1]] = -1 + 0j
    C_m = np.matrix(C_m)    
    c_b = np.zeros([eq_cons_ind.shape[1],1],dtype=np.complex_)
    h_est = np.zeros_like(h_act, dtype=np.complex_)
    error = 0
    pbar = tqdm(range(trials))
    for trial in pbar:
        xx = get_random_xx()
        X = get_X(xx)
        F = get_F()
        y = get_y(X,F,h_act)
        A = get_A(X,F)
        h_est_tt = get_estim_cons_h(A,y,C_m,c_b)
        error_tt = np.linalg.norm(h_act - h_est_tt)
        h_est += (h_est_tt - h_est)/(trial + 1)
        error += (error_tt - error)/(trial + 1)
        pbar.set_description(f"Trial {trial} error {error}")
    
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {error}")
    return h_act, h_est

@ex.capture
def plot_h(h_true, h_estim):
    h_true_mag = np.absolute(h_true)
    h_estim_mag = np.absolute(h_estim)
    plt.plot(h_true_mag, 'o', label='True h')
    plt.plot(h_estim_mag, 'o', label='Estimated h')
    plt.legend(loc='best')
    plt.show()

@ex.automain
def main():
    h_act, h_est = q3()
    plot_h(h_act,h_est)