import numpy as np
from scipy.optimize import least_squares
import math


def multi_exp_wrapper(b, signal, peaks):
    def multi_exp(x):
        # Define NLLS multi-exp fitting function

        f = np.array(0)
        for i in range(peaks - 2):
            f = f + np.array(math.exp(-np.kron(b, abs(x[i]))) * x[peaks + i])
        return (
            f
            + np.array(
                math.exp(-np.kron(b, abs(x[peaks - 1]))) * (100 - (np.sum(x[peaks:-1])))
            )
            - signal
        )  # fun needs to return an array of shape [peaks,]?!

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
    # if called w\o dIn and fIn uses default tri-exp start values for dIn and fIn [Periquito2021]

    peaks = 3
    d = f = np.zeros((len(signal), len(signal), peaks))

    for i in range(len(signal)):
        for j in range(len(signal)):

            # Declare start values
            x0 = np.append(dIn[i][j][:], fIn[i][j][:-1])

            # Number of found compartments by NNLS
            peaks = np.count_nonzero(x0[0:2])

            # TODO: bounds neccessary?
            lb = [
                np.repeat(Dmin, peaks),
                np.repeat(0, peaks - 1),
            ]
            ub = [np.repeat(Dmax, peaks), np.repeat(100, peaks - 1)]

            # Scale signal for NLLS to find reasonable volume fractions
            scaling = 100 / signal[i][j][1]
            scaledSignal = np.multiply(signal[i][j][:], scaling)

            x0 = np.array([1.35 * 1e-3, 4 * 1e-3, 52.5])
            # NLLS fitting result containing d and f
            result = least_squares(
                multi_exp_wrapper(b, scaledSignal, peaks),
                x0,
                bounds=(lb, ub),
                method="lm",
            )
            s = result.x
            s = np.append(s, 100 - sum(s[peaks:-1]))

            d[i][j][:] = s[0 : peaks - 1]
            f[i][j][:] = s[peaks:-1]

    return d, f
