import math
import numpy as np
from scipy.optimize import least_squares

def monoExp(x, b, signal):
    return np.array(math.exp(-math.kron(b, abs(x(1)))) - signal)

def biExp(x, b, signal):
    return np.array(math.exp(-math.kron(b, abs(x(1))))*x(4) + math.exp(-math.kron(b, abs(x(2))))*(100-x(4)) - signal)

def triExp(x, b, signal):
    return math.exp(-math.kron(b, abs(x(1))))*x(4) + math.exp(-math.kron(b, abs(x(2))))*x(5) + math.exp(-math.kron(b, abs(x(3))))*(100-(x(4)+x(5))) - signal

def quadExp(x, b, signal):
    return np.array(math.exp(-math.kron(b, abs(x(1))))*x(5) + math.exp(-math.kron(b, abs(x(2))))*x(6) + math.exp(-math.kron(b, abs(x(3))))*x(7) + math.exp(-math.kron(b, abs(x(4))))*(100-(x(5)+x(6)+x(7))) - signal)


def NLLSfitting(b, signal, Dmin, Dmax, dIn=[1.35*1e-3, 4*1e-3, 155*1e-3], fIn=[52.5, 40]):
# NLLSfitting(inputSimu, dIn, fIn) =  a priori information dNNLS and fNNLS in x0
# NLLSfitting(inputSimu) = no a priori information, using standard start value
# default tri-exp start values for dIn and fIn [Periquito2021]

    input = [dIn, fIn].T
    x0 = input[1:-2]
    
    np = np.count_nonzero(x0[1:3])                      # number of found compartments by NNLS   

    lb = [np.repeat(Dmin,np), np.repeat(0,np-1)]        # set bound constraints based on NNLS d range
    ub = [np.repeat(Dmax,np), np.repeat(100,np-1)]      # TODO: bounds neccessary?
    
    scaling = 100/signal(1)                             # scale signal for NLLS to find reasonable volume fractions
    signal = np.multiply(signal,scaling)
        
    if     np == 3:
        # Create tri-exponential signal function for fitting with d and f as fitting variable
        result = least_squares(triExp(x0, b, signal), x0, bounds=(lb, ub), method='lm')
        s = result.x
        s[6] =100-(s[4]+s[5])

    elif np == 2:
        # Create bi-exponential signal function
        result = least_squares(biExp(x0, b, signal), x0[1:4],[],[], method='lm') 
        s = result.x
        s[5] = 100-s(4)
        s[6] = 0

    elif np == 1:
        # Create mono-exponential signal function
        result = least_squares(monoExp, x0(1),[],[], method='lm') 
        s[1] = result.x
        s[2:6] = np.zeros(1,5)
        s[4] = 1

    elif np == 4:
        # Create 4-exponential signal function
        result = least_squares(quadExp, x0, bounds=(lb, ub), method='lm')
        s = result.x
        s[8] = 100-sum(s[5:8])

    d = abs(s[1:np])
    f = s[np+1:-1]
    return d, f, result