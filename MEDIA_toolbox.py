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
# TODO: parrallelise code

# Running NNLS simulations
sNNLSNoReg, sNNLSReg, mu = NNLSfitting(DBasis, signal)

# Calculating NNLS diffusion parmeters (0 = noReg, 1 = Reg)
dNNLS, fNNLS = findpeaksNNLS(sNNLSReg, DValues)

# NLLS/ tri-exponential with NNLS results as a priori information
# TODO: Adjust code for pixelwise analysis
# TODO: fix inaccurate NLLS results
dNLLS, fNLLS, resnormNLLS = NLLSfitting(b, signal, Dmin, Dmax, dNNLS, fNNLS.T)

# TODO: additional structures to save relevant results?

###########################################################################
# Plotting and saving data
###########################################################################

# Plotting figures
# TODO: Adjust plotSimu to maps instead of mean signal
plotSimu(signal, b, fitNNLS)

# Write simulation data to file
filename = ["NNLSfitting.txt" "MEDIAresults.txt"]
