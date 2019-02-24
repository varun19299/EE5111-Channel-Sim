import numpy as np
from sacred import Experiment
import cmath
from tqdm import tqdm

ex = Experiment("simulation")

@ex.config
def config():
    x_ii = np.array([1 + 1j,-1 + 1j,1 - 1j,-1 - 1j], dtype= np.complex_)    # diag of X belongs to
    N = 1024                          # no of freqs
    n = 16                            # no of channel taps
    trials = 10000                    # no of trials per estimation

    sigma_real = 0.1
    sigma_imag = 0.1

    d = 0.2                           # Decay factor
    guard = 0                         # Guard Bands

@ex.capture
def get_F(N, n):
    '''
    Generate IDFT Matrix
    : N : No of frequencies
    : n : channel taps 
    
    Returns:
    Float Matrix: N*n
    '''
    F = np.zeros((N,n), dtype = np.complex_) 
    for i in range(N):
        for j in range(n):
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
def get_h(n, d=0.2, sigma2_real=0.5, sigma2_imag=0.5):
    '''
    n tap time domain channel vector

    Generates h vec according to exponentially decaying power-delay profile
    : sigma_real : variance of zero mean gaussian dist for real part
    : sigma_imag : variance of zero mean gaussian dist for imag part
    '''
    a = np.random.normal(0, np.sqrt(sigma2_real), size = n)
    b = np.random.normal(0, np.sqrt(sigma2_imag), size = n)
    p = np.exp(-d*np.arange(n))
    h = p * (a + b*1j)
    h = h/np.linalg.norm(p)
    return h[:,None]

@ex.capture
def get_h_sparse(n, zero_ind,d=0.2, sigma2_real=0.5, sigma2_imag=0.5):
    '''
    n tap time domain channel vector

    Generates h vec according to exponentially decaying power-delay profile
    : sigma_real : variance of zero mean gaussian dist for real part
    : sigma_imag : variance of zero mean gaussian dist for imag part
    '''
    h = get_h(n,d,sigma2_real,sigma2_imag)
    h[zero_ind] = 0 + 0j
    return h

@ex.capture
def get_y(X,F,h,sigma2=0.1,sigma2_real=0.05):
    '''
    Generates y vector

    Returns: 
    : y of size N 
    : h_actual of size n
    '''
    assert (sigma2>sigma2_real)
    y = np.matmul(np.matmul(X,F),h)
    N = y.shape[0]
    y = y + (np.random.normal(0,np.sqrt(sigma2_real),size=N) + np.random.normal(0,np.sqrt(sigma2-sigma2_real),size=N) * 1j)[:,None]
    return y

@ex.capture
def get_A(X,F):
    return np.matrix(np.matmul(X,F))

@ex.capture
def get_estim_h(A,y,lam=0):
    '''

    '''
    h_est = np.linalg.inv(np.matmul(A.getH(),A)+lam*(1+0j)*np.eye(A.shape[1]))
    h_est = np.matmul(h_est,A.getH())
    h_est = np.matmul(h_est,y)
    return h_est

@ex.capture
def get_estim_sparse_h(A,y,zero_ind,lam=0):
    h_est = get_estim_h(A,y)
    L = h_est.shape[0]
    r = len(zero_ind)
    C_m = np.eye(L)
    C_m = np.matrix(C_m[zero_ind,:])
    c_b = np.zeros([r,1])
    tempM = np.matmul(np.linalg.inv(np.matmul(A.getH(),A)+lam*(1+0j)*np.eye(A.shape[1])),C_m.getH())
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
def q1(guard, N, trials):
    '''
    LSE of h_act for 10,000 trials
    '''
    print("\n\n -----\n Question 1\n LSE of h_act for 10,000 trials")
    h_act = get_h()
    h_est = np.zeros_like(h_act, dtype=np.complex_)
    error = 0

    pbar = tqdm(range(trials))
    for trial in pbar:
        xx = get_random_xx()[:1024 - guard*2]

        X = get_X(xx, N=N, guard=guard)
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

@ex.capture
def q3(guard, N, x_ii, L, trials):
    '''
    200 zero guard bands
    '''
    print("\n\n -----\n Question 3\n 200 zero guard bands, each side")
    non_zero_ind = [2,5,7,9,10,12]
    zero_ind = np.delete(np.arange(L),non_zero_ind)
    h_act = get_h_sparse(L,zero_ind)
    error = 0
    h_est = np.zeros_like(h_act, dtype=np.complex_)

    pbar = tqdm(range(trials))
    for trail in pbar:
        xx = get_random_xx()[:N - 2*guard]
        X = get_X(xx, N,guard=guard)
        
        F = get_F()
        y = get_y(X,F,h_act)
        A = get_A(X,F)

        h_est_tt = get_estim_sparse_h(A,y,zero_ind,0.5)
        error = np.linalg.norm(h_act - h_est_tt)

        h_est += (h_est_tt - h_est)/(trial + 1)
        error += (error_tt - error)/(trial + 1)

        pbar.set_description(f"Trial {trial} error {error}")
    
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {error}")

@ex.automain
def main():
    q1()
    # q2()
    # q3()

    