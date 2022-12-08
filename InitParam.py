# Initial parameters are set here

import numpy as np
import nibabel as nib

# Initialize variables
files = ["./testData/scans4Db16.nii", "./testData/segmentationC.nii"]
nii = nib.load(files[0])
nii = np.array(nii.get_data())
nii = np.squeeze(nii)  # drop dim=1, get 3D array [px px b-values]
ROI = np.array(nib.load(files[1]).get_data())
b = np.array([[0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 525, 750]])

# Generate basis values
Dmin, Dmax = 0.7 * 1e-3, 300 * 1e-3  # [vanBaalen2016] & [Wong2019]
m = 300  # no. of bins
DValues = np.array(np.logspace(np.log10(Dmin), np.log10(Dmax), m))
DBasis = np.exp(-np.kron(b.T, DValues))
