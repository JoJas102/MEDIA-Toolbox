import math
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_widths


def findpeaksNNLS(s, DValues):
    # find peaks and diffusion coefficients of NNLS fitting results

    # Matrix indices at iteration i
    # [1,i] slow/tissue
    # [2,i] inter/tubular
    # [3,i] fast/blood

    # [maxima,dNNLS,widths,proms] = findpeaks(s, 1:length(DValues), 'SortStr', 'descend', 'WidthReference','halfheight') # returns maxima, x positions, FWHM and prominences of peaks

    d, properties = find_peaks(s)  # TODO: descending output?!
    fwhm, _ = peak_widths(s, d, rel_height=0.5)
    maxima = properties.peak_heights

    d = DValues(d)  # convert back to log scale values

    # Calc area under gaussian curve
    f = (
        np.multiply(maxima, fwhm)
        / (2 * math.sqrt(2 * math.log(2)))
        * math.sqrt(2 * math.pi)
    )

    # Threshold
    d[f < 0.03] = 0  # remove any entry with vol frac lower 3%
    # TODO: obsolet code if threshold/prominence adjusted in find_peaks
    f[f < 0.03] = 0

    f = np.divide(f, sum(f)) * 100  # normalize f

    result = [maxima, np.transpose(d), np.transpose(fwhm), f]
    return d, f, result
