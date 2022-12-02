from scipy.optimize import nnls


def NNLSfitting(DBasis, signal):

    for i in len(signal):
        for j in len(signal):
            # NNLS fitting w\ reg minimises norm(A*s-signal) for reference
            s = nnls(DBasis, signal[i][j][:])

            # NNLS fitting w reg
            [sReg, mu, resid] = CVNNLS(DBasis, signal[i][j][:])  # CVNNLS from Bjarnason
            # larger mu = more satisfaction of constraints at expanse of increasing misfit (Witthal1989)

    return s, sReg, mu
