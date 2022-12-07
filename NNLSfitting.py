import regNNLS
import numpy as np
from scipy.optimize import nnls


def NNLSfitting(DBasis, signal):

    s, sReg = np.zeros(len(signal), len(signal), len(DBasis))

    for i in len(signal):
        for j in len(signal):
            # NNLS fitting w\ reg minimises norm(A*s-signal) for reference
            s[i][j][:] = nnls(DBasis, signal[i][j][:])

            # NNLS fitting w reg (CVNNLS from Bjarnason)
            sReg[i][j][:], mu, resid = regNNLS(DBasis, signal[i][j][:])
            # larger mu = more satisfaction of constraints at expanse of increasing misfit (Witthal1989)

    return s, sReg, mu
