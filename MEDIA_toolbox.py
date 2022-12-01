###########################################################################
#
#           Multi-Exponential Decay Image Analysis (MEDIA) Toolbox
#
###########################################################################
# Init parameters
###########################################################################

import numpy as np
import NNLSfitting, findpeaksNNLS, NLLSfitting, plotSimu
from InitParam import b, nii, ROI, DValues, DBasis, Dmin, Dmax

###########################################################################
# Read NIfTIs, cut out ROIs and extract mean signal
###########################################################################

# TODO: test cutting ROI and extracting signal from NIFTIs
signal = np.multiply(nii, np.tile(ROI.img, ([1, 1, len(b)])))
meanSignal = np.squeeze(np.mean(signal)).T

###########################################################################
# Signal analysis
###########################################################################

# Running NNLS simulations
sNNLSNoReg, sNNLSReg, mu = NNLSfitting(DBasis, meanSignal.T)
# TODO: check dimension
fitNNLS = [
    DValues,
    sNNLSNoReg.T,
    sNNLSReg.T,
    [mu, np.zeros(1, len(DValues) - 1)],
]

# Calculating NNLS diffusion parmeters (0 = noReg, 1 = Reg)
dNNLS, fNNLS, results = findpeaksNNLS(fitNNLS, 1)

# NLLS/ tri-exponential with NNLS results as a priori information
# TODO: fix inaccurate NLLS results
dNLLS, fNLLS, resnormNLLS = NLLSfitting(b, meanSignal, Dmin, Dmax, dNNLS, fNNLS.T)
results[:, 5:7] = [dNLLS.T, fNLLS.T, [resnormNLLS, 0, 0].T]


###########################################################################
# Plotting and saving data
###########################################################################

# Plotting figures
plotSimu(meanSignal, b, fitNNLS)

# Write simulation data to file
filename = ["NNLSfitting.txt" "MEDIAresults.txt"]
# write3DMatrixToTxt(fitNNLS, filename(1))
# write3DMatrixToTxt(results, filename(2))
