# MEDIA-Toolbox

## Description

Program to analyse multi-exponential signal compartments in DWI images/scans with several b-values using NNLS fitting among others to fit diffusion data and compare multiple multi-exponential fitting methods. Finding the total number of components contributing to the multi-exponential signal and analysing the results by calculating corresponding diffusion parameters.
This fitting routine uses the regularized NNLS algorithm with cross validation (CVNNLS.m) from Thorarin Bjarnason.

Generally capital letters represent matrices or tensors and small letters stand for variuos variables, vectors and constants.

### Functions
To carry out this simulation the following functions need to be located in the path:
* InitParam.py
* MEDIA_toolbox.py
* NNLSfitting.py
* NLLSfitting.py
* CVNNLS.py
* findpeaksNNLS.py
* plotSimu.py

### Initial parameters neccessary
In [InitParam.py](InitParam.py) essential simulation and acquisition parameters are set:
* files[] = string array containing filepaths to scan image (4D NIfTI) and ROI (segmentation containing region of interest as NIfTI)
* b[] = list of used b-values
* Dmin and Dmax = fitting range to calculate DValues for fitting 
* m = number of bins for fitting procedure endregion

For any further comments or explaining descriptions see annotations
inside the functions code files.

## General coding information
Requires Python 3.11
Formatting is taken care of by black


Jonas Jasse
Last modified 01.12.2022
