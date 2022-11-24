function [dNNLS, fNNLS, resultNNLS] = findpeaksNNLS(matrix, reg)
    
    # statistics for fit with (reg = 1) and w\ regression (0)
    if reg == 1
        s(:,:) =  matrix(3,:,:) # sNNLSReg
    else
        s(:,:) =  matrix(2,:,:) # sNNLSNoReg
    end
    
    s = s' # right array dimensions
    DValues =  matrix(1,:,1)
    
    # Matrix indices at iteration i
    # [1,i] slow/tissue
    # [2,i] inter/tubular
    # [3,i] fast/blood

    [maxima,dNNLS,widths,proms] = findpeaks(s, 1:length(DValues), 'SortStr', 'descend', 'WidthReference','halfheight') # returns maxima, x positions, FWHM and prominences of peaks
    dNNLS = DValues(dNNLS)                                                              # convert back to log scale values
    #findpeaks(s, 1:length(DValues), 'Annotate','extents', 'SortStr', 'descend', 'NPeaks', 3) # check findpeaks fct visually

    # Calc area under gaussian curve
    fNNLS = maxima(:).*widths(:)/(2*sqrt(2*log(2)))*sqrt(2*pi)

    # Threshold 
#     dNNLS(fNNLS<0.01)=0 # remove any entry with vol frac lower 1#
#     fNNLS(fNNLS<0.01)=0
    fNNLS = fNNLS./sum(fNNLS)*100 # normalize f
    
    resultNNLS = [maxima, dNNLS', widths', fNNLS]
end