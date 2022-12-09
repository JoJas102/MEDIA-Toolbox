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

    dim = 4
    d = np.zeros((len(signal), len(signal), dim))
    f = np.zeros((len(signal), len(signal), dim))

    for i in range(len(signal)):
        for j in range(len(signal)):

            # Skipping pixels not included in ROI
            if np.sum(signal[i][j][:]) == 0:
                break

            # Scale signal for NLLS to find reasonable volume fractions
            scaling = 100 / signal[i][j][1]
            scaledSignal = np.multiply(signal[i][j][:], scaling)

            # Declare start values (NNLS results)
            x0 = np.append(dIn[i][j][:], fIn[i][j][:-1])

            # Number of found compartments (by NNLS)
            peaks = np.count_nonzero(x0[0:2])

            # NLLS fitting result containing d and f
            result = least_squares(
                multi_exp_wrapper(b, scaledSignal, peaks), x0, method="lm"
            )

            dOut = result.x[:peaks]
            fOut = np.append(result[peaks:], 100 - sum(result[peaks:]))

            # Fill with zeros for uniform dimension
            if len(dOut) < dim:
                dOut = np.append(dOut, np.zeros(dim - len(dOut)))
                fOut = np.append(fOut, np.zeros(dim - len(fOut)))

            d[i][j][:] = dOut
            f[i][j][:] = fOut

    return d, f
