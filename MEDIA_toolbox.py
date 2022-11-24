###########################################################################
#
#           Multi-Exponential Decay Image Analysis (MEDIA) Toolbox 
#{
###########################################################################
# General information
###########################################################################
# 
# Program to analyse multi-exponential signal compartments in DWI images/scans 
# with several b-values using NNLS fitting of the diffusion data and compare 
# multi-exponential fitting methods. Finding the total number of components
# contributing to the corresponding multi-exponential signal and analysing 
# the results by calculating corresponding diffusion parameters.
# This fitting routine uses the regularized NNLS algorithm with cross 
# validation (CVNNLS.m) from Thorarin Bjarnason for comparison.
# 
# Generally capital letters represent matrices or tensors and small letters
# stand for variuos variables, vectors and constants.
#
# To carry out this simulation the following functions need to be located
# in the MATLAB path:
# InitVar.m
# load_nii.m (NIfTI package) 
# NNLSfitting.m
# CVNNLS.m
# fastnnls.m
# findpeaksNNLS.m
# plotSimu.m 
# write3DMatrixToTxt.m 
#
# For any further comments or explaining descriptions see annotations
# inside the functions code files.
#
# - 
# Jonas Jasse 
# Last modified 18.04.2022
#
###########################################################################
# Initial variables (InitVar.m)
###########################################################################
#
# - filepaths{} = cell array containing filepaths to scan image (4D NIfTI) 
#   and ROI (segmentation containing region of interest as NIfTI)
# - b = list of used b-values for plotting
# - DValues = fitting range
# - m = number of bins for fitting procedure
#}
###########################################################################
# Init parameters
###########################################################################

clear
[b, nii, ROI, DValues, DBasis, Dmin, Dmax] = InitVar();

###########################################################################
# Read NIfTIs, cut out ROIs and extract mean signal
###########################################################################

signal = nii.*repmat(double(ROI.img),[1,1,length(b)]); 
meanSignal = squeeze(mean(signal,1:2))'; 

###########################################################################
# Signal analysis
###########################################################################

# Running NNLS simulations
[sNNLSNoReg, sNNLSReg, mu] = NNLSfitting(DBasis, meanSignal'); 
fitNNLS = [DValues; sNNLSNoReg'; sNNLSReg'; [mu zeros(1,length(DValues)-1)]]; 

# Calculating NNLS diffusion parmeters (0 = noReg, 1 = Reg)
[dNNLS, fNNLS, results] = findpeaksNNLS(fitNNLS, 1);

# NLLS/ tri-exponential with NNLS results as a priori information
[dNLLS, fNLLS, resnormNLLS]  = NLLSfitting(b, meanSignal, dNNLS, fNNLS', Dmin, Dmax); #TODO: fix inaccurate NLLS results
results(:,5:7) = [dNLLS' fNLLS' [resnormNLLS 0 0]' ];


###########################################################################
# Plotting and saving data
###########################################################################

# Plotting figures
plotSimu(meanSignal, b, fitNNLS); #TODO: hier weiter

# Write simulation data to file
filename = ["NNLSfitting.txt" "MEDIAresults.txt"];
write3DMatrixToTxt(fitNNLS, filename(1)); 
write3DMatrixToTxt(results, filename(2));