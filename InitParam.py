# Initial parameters are set here

import math
import numpy as np
import nibabel as nib

# Initialize variables
files = ['.\test data\scans4Db16.nii' '.\test data\segmentationC.nii']
nii = nib.load(files[1])
nii = np.squeeze(nii)
ROI = nib.load(files[2])
b = [0,5,10,20,30,40,50,75,100,150,200,250,300,400,525,750]

# Generate basis values
Dmin = 0.7*1e-3       # [vanBaalen2016] & [Wong2019]
Dmax = 300*1e-3
m = 300               # no. of bins             
DValues = np.logspace(np.log10(Dmin), np.log10(Dmax), m) 
DBasis = math.exp(-math.kron(b.T), DValues))
