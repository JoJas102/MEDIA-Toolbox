# Initial parameters are set here

    format long

    # Initialize variables
    addpath('.\NIfTI_20140122')
    filepath = {'.\test data\scans4Db16.nii' '.\test data\segmentationC.nii'}
    nii = load_nii(filepath{1}) # load nii as double matrix with dim neq 1
    nii = squeeze(double(nii.img))
    ROI = load_nii(filepath{2})
    b = [0,5,10,20,30,40,50,75,100,150,200,250,300,400,525,750]

    # Generate basis values
    Dmin = 0.7*1e-3       # [vanBaalen2016] & [Wong2019]
    Dmax = 300*1e-3
    m = 300               # no. of bins             
    DValues = logspace(log10(Dmin), log10(Dmax), m) 
    DBasis = exp(-kron( b', DValues))

    # Define total diffusion signal decay according to sum of f_i*e^(-b*D_i)
    #A = exp(-kron(b', d)) # constraint matrix containing exp decay funxtions
    #rawSignal = A*f'      # mean signal of ROI/voxel without noise