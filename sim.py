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
def get_h(n, d, sigma_real, sigma_imag):
    '''
    n tap time domain channel vector

    Generates h vec according to exponentially decaying power-delay profile
    : sigma_real : variance of zero mean gaussian dist for real part
    : sigma_imag : variance of zero mean gaussian dist for imag part
    '''
    a = np.random.normal(0, np.sqrt(1/2), size = n)
    b = np.random.normal(0, np.sqrt(1/2), size = n)
    h = np.exp(-d*np.arange(n)) * (a + b*1j)
    h = h/np.linalg.norm(h)
    return h

@ex.capture
def get_y():
    '''
    Generates y vector

    Returns: 
    : y of size N 
    : h_actual of size n
    '''
    h = get_h()
    F = get_F()
    X = get_X()
    return np.matmul(np.matmul(X,F),h), h

@ex.automain
def main():
    y, h_act = get_y()
    X = get_X()
    F = get_F()
    A = np.matmul(X,F)      # usual y = Ah + n
    h_est = np.linalg.inv(np.matmul(A.T,A))
    h_est = np.matmul(h_est,A.T)
    h_est = np.matmul(h_est, y)
    error = h_act - h_est
    print(f"h actual {h_act}")
    print(f"h estimated {h_est}")
    print(f"Error L2 in linear estimation is {np.linalg.norm(error)}")