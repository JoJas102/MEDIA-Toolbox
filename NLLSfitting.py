import numpy as np
from scipy.optimize import least_squares
import math

def multi_exp_wrapper(b, signal, np):

    def multi_exp(x):
        f = np.array(0)
        for i in range(np - 2):
            f = f + np.array(math.exp(-np.kron(b, abs(x(i)))) * x(np + i))
        return f + np.array(math.exp(-np.kron(b, abs(x(np - 1)))) * (100 - (np.sum(x[np:-1])))) - signal
    return multi_exp

def NLLSfitting(
    b, signal, Dmin, Dmax, dIn=[1.35 * 1e-3, 4 * 1e-3, 155 * 1e-3], fIn=[52.5, 40]
):
    # NLLSfitting(inputSimu, dIn, fIn) =  a priori information dNNLS and fNNLS in x0
    # NLLSfitting(inputSimu) = no a priori information, using standard start value
    # default tri-exp start values for dIn and fIn [Periquito2021]

    d, f = np.zeros(len(signal), len(signal), len(3))

    for i in len(signal):
        for j in len(signal):

            input = [dIn, fIn].T
            x0 = input[0:-2]

            np = np.count_nonzero(x0[0:2])  # number of found compartments by NNLS

            # TODO: bounds neccessary?
            lb = [
                np.repeat(Dmin, np),
                np.repeat(0, np - 1),
            ]
            ub = [np.repeat(Dmax, np), np.repeat(100, np - 1)]

            # Scale signal for NLLS to find reasonable volume fractions
            scaling = 100 / signal[i][j][1]
            scaledSignal = np.multiply(signal[i][j][:], scaling)

            result = least_squares(multi_exp_wrapper(b, scaledSignal, np), x0, bounds=(lb, ub), method="lm")
            s = result.x
            s = np.append(s, 100 - sum(s[np:-1]))

            d[i][j][:] = abs(s[1:np])
            f[i][j][:] = s[np + 1 : -1]

    return d, f
