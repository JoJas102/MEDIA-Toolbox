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
# RCut out ROIs and extract signal
###########################################################################

signal = np.multiply(nii, ROI)

###########################################################################
# Signal analysis
###########################################################################

# Running NNLS simulations
# TODO: write NNLS code for full array instead of meanSignal array only
# TODO: parrallelise code
sNNLSNoReg, sNNLSReg, mu = NNLSfitting(DBasis, signal)

# Calculating NNLS diffusion parmeters (0 = noReg, 1 = Reg)
dNNLS, fNNLS, results = findpeaksNNLS(sNNLSReg, DValues)

# NLLS/ tri-exponential with NNLS results as a priori information
# TODO: write NLLS code for full array instead of meanSignal array only
# TODO: fix inaccurate NLLS results
dNLLS, fNLLS, resnormNLLS = NLLSfitting(b, signal, Dmin, Dmax, dNNLS, fNNLS.T)
results[:, 5:7] = [dNLLS.T, fNLLS.T, [resnormNLLS, 0, 0].T]

# TODO: create structures to save relevant results

###########################################################################
# Plotting and saving data
###########################################################################

# Plotting figures
# TODO: Adjust plotSimu to maps instead of mean signal
plotSimu(signal, b, fitNNLS)

# Write simulation data to file
filename = ["NNLSfitting.txt" "MEDIAresults.txt"]
