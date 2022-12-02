import math
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_widths


def findpeaksNNLS(s, DValues):
    # find peaks and diffusion coefficients of NNLS fitting results

    for i in len(s):
        for j in len(s):

            # TODO: descending output?!
            # TODO: thresholding possible?
            d, properties = find_peaks(s[i][j][:])
            fwhm, _ = peak_widths(s[i][j][:], d, rel_height=0.5)
            maxima = properties.peak_heights

            # Convert back to log scale values
            d = DValues(d)

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
