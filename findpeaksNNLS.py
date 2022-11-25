import math
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_widths

def findpeaksNNLS(matrix, reg):
    
    # statistics for fit with (reg = 1) and w\ regression (0)
    if reg == 1:
        s =  matrix[3,:,:]                   # sNNLSReg
    else:
        s =  matrix[2,:,:]                   # sNNLSNoReg
    
    s = np.transpose(s)                      # right array dimensions
    DValues =  matrix[1,:,1]
    
    # Matrix indices at iteration i
    # [1,i] slow/tissue
    # [2,i] inter/tubular
    # [3,i] fast/blood

    #[maxima,dNNLS,widths,proms] = findpeaks(s, 1:length(DValues), 'SortStr', 'descend', 'WidthReference','halfheight') # returns maxima, x positions, FWHM and prominences of peaks

    dNNLS, properties = find_peaks(s)  # TODO: descending output?! 
    fwhm, _ = peak_widths(s, dNNLS, rel_height=0.5)
    maxima = properties.peak_heights

    dNNLS = DValues(dNNLS)                   # convert back to log scale values

    # Calc area under gaussian curve
    fNNLS = np.multiply(maxima,fwhm)/(2*math.sqrt(2*math.log(2)))*math.sqrt(2*math.pi)

    # Threshold 
    dNNLS[fNNLS<0.03] = 0                    # remove any entry with vol frac lower 3%
    fNNLS[fNNLS<0.03] = 0                    # TODO: obsolet code if threshold/prominence adjusted in find_peaks
    
    fNNLS = np.divide(fNNLS,sum(fNNLS))*100  # normalize f
    
    resultNNLS = [maxima, np.transpose(dNNLS), np.transpose(fwhm), fNNLS]
    return dNNLS, fNNLS, resultNNLS