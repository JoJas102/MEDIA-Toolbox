import numpy as np
from scipy.optimize import least_squares
import math


def multi_exp_wrapper(b, signal, peaks):
    def multi_exp(x):
        # Define NLLS multi-exp fitting function

        f = 0
        for i in range(peaks - 2):
            f = +np.exp(-np.kron(b, abs(x[i]))) * x[peaks + i]
        f = +np.exp(-np.kron(b, abs(x[peaks - 1]))) * (100 - (np.sum(x[peaks:])))
        return f - signal

    return multi_exp


def NLLSfitting(
    b,
    signal,
    Dmin,
    Dmax,
    dIn=np.tile([1.35 * 1e-3, 4 * 1e-3, 155 * 1e-3], (300, 300, 1)),
    fIn=np.tile([52.5, 40, 7.5], (300, 300, 1)),
):
    # NLLS fitting routine
    # if called w\o dIn and fIn uses default tri-exp start valuesfor d and f [Periquito2021]

    # TODO: how many peaks/entrys for array initilisation?
    peaks = 2
    d = np.zeros((len(signal), len(signal), peaks))
    f = np.zeros((len(signal), len(signal), peaks))
    for i in range(len(signal)):
        for j in range(len(signal)):

            # Declare start values
            x0 = np.append(dIn[i][j][:], fIn[i][j][:-1])

            # Number of found compartments by NNLS
            peaks = np.count_nonzero(x0[0:2])

            # Scale signal for NLLS to find reasonable volume fractions
            scaling = 100 / signal[i][j][1]
            scaledSignal = np.multiply(signal[i][j][:], scaling)

            x0 = np.array([1.35 * 1e-3, 4 * 1e-3, 52.5])
            scaledSignal = np.array(
                [
                    100,
                    90,
                    85,
                    80,
                    75,
                    70,
                    60,
                    55,
                    50,
                    45,
                    40,
                    30,
                    20,
                    10,
                    5,
                    2,
                ]
            )

            # NLLS fitting result containing d and f
            result = least_squares(
                multi_exp_wrapper(b, scaledSignal, peaks), x0, method="lm"
            )
            s = result.x
            s = np.append(s, 100 - sum(s[peaks:]))

            d[i][j][:] = s[0:2]
            f[i][j][:] = s[peaks:]

    return d, f
