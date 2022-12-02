import math
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_widths


def findpeaksNNLS(s, DValues):
    # find peaks and diffusion coefficients of NNLS fitting results

    d_tot, f_tot = np.zeros(len(s), len(s), len(3))

    for i in len(s):
        for j in len(s):

            # TODO: descending output?!
            # TODO: thresholding possible?
            # TODO: find how many peaks?
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

            # Threshold (remove entries with vol frac < 3%)
            d[f < 0.03] = 0
            # TODO: obsolet code if threshold/prominence adjusted in find_peaks
            f[f < 0.03] = 0

            f = np.divide(f, sum(f)) * 100  # normalize f

            d_tot[i][j][:] = d
            f_tot[i][j][:] = f

    return d_tot, f_tot
