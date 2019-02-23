import numpy as np
from sacred import Experiment
import cmath

ex = Experiment("simulation")

@ex.config
def config():
    x_ii = np.array([1 + 1j,-1 + 1j,1 - 1j,-1 - 1j], dtype= np.complex_)    # diag of X belongs to
    N = 1024                          # no of freqs
    n = 16                            # no of channel taps

    sigma_real = 0.1
    sigma_imag = 0.1

    d = 0.2                           # Decay factor

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
def get_X(N, x_ii):
    '''
    Generates diagonal matrix with known symbols
    '''
    assert not (N%4), "N must be divisible by 4"

    xx = np.tile(x_ii, N//4)
    X = np.diag(xx)
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
    return h

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
def get_y(X,F,h):
    '''
    Generates y vector

    Returns: 
    : y of size N 
    : h_actual of size n
    '''
    return np.matmul(np.matmul(X,F),h)

@ex.capture
def get_A(X,F):
    return np.matrix(np.matmul(X,F))

@ex.capture
def get_estim_h(A,y):
    h_est = np.linalg.inv(np.matmul(A.getH(),A))
    h_est = np.matmul(h_est,A.getH())
    h_est = np.matmul(h_est, y)
    return h_est

@ex.capture
def get_estim_sparse_h(A,y,zero_ind):
    h_est = get_estim_h(A,y)
    L = h_est.shape[1]
    r = len(zero_ind)
    C_m = np.eye(L)
    C_m = np.matrix(C_m[zero_ind,:])
    c_b = np.zeros([r,1])
    tempM = np.matmul(np.linalg.inv(np.matmul(A.getH(),A)),C_m.getH())
    constraint = np.matmul(C_m,h_est.T) - c_b
    h_sparse_estim = h_est - np.matmul(np.matmul(tempM,np.linalg.inv(np.matmul(C_m,tempM))),constraint).T
    return h_sparse_estim


@ex.automain
def main():
    N = 1024
    L = 16
    x_ii = np.array([1+1j,-1+1j,1-1j,-1-1j])
    X = get_X(N,x_ii)
    non_zero_ind = [2,5,7,9,10,12]
    zero_ind = np.delete(np.arange(L),non_zero_ind)
    h_act = get_h_sparse(L,zero_ind)
    F = get_F(N,L)
    y = get_y(X,F,h_act)
    A = get_A(X,F)
    h_est = get_estim_sparse_h(A,y,zero_ind)    
    error = h_act - h_est
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {np.linalg.norm(error)}")