from regNNLS import *
import numpy as np
from scipy.optimize import nnls


def NNLSfitting(DBasis, signal):

    s = np.zeros((len(signal), len(signal), len(DBasis[1][:])))
    sReg = np.zeros((len(signal), len(signal), len(DBasis[1][:])))

    for i in range(len(signal)):
        for j in range(len(signal)):

            # Skipping pixels not included in ROI
            if np.sum(signal[i][j][:]) == 0:
                continue

            # TODO: find solution for nnls Runtiome error: too many iterations
            # NNLS fitting w\ reg minimises norm(A*s-signal) for reference
            s[i][j][:], _ = nnls(DBasis, signal[i][j][:])

            # NNLS fitting w reg (CVNNLS from Bjarnason)
            sReg[i][j][:], mu, resid = regNNLS(DBasis, signal[i][j][:])
            # larger mu = more satisfaction of constraints at expanse of increasing misfit (Witthal1989)

    return s, sReg
