import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plotSimu(signal, b, fitNNLS):

    DValues =  fitNNLS[1,:]
    s =  fitNNLS[2,:]
    s2 =  fitNNLS[3,:]
    DBasis = math.exp(-np.kron(b.T, DValues))
    
    plt.figure(1)
    singlePlot(signal, DBasis, b, DValues, s, s2)
    return

def singlePlot(signal, DBasis, b, DValues, s, s2):

    plt.subplot(311)
    y_recon = DBasis*s.T
    y_recon2 = DBasis*s2.T
    plt.plot( b , signal , 'ko' , b , y_recon , 'b-', b, y_recon2 , 'r-') # compare signal and fitting result
    plt.ylim([0, signal(1)])
    plt.title('(a) Signal decay fitting')
    plt.xlabel('b-values (s/mm^2)')
    plt.ylabel('Signal amplitude')
    plt.xlim([0, b[-1]])
    plt.legend('data (a), D^{in}_i (c)','lsqnonneg','NNLSreg')

    plt.subplot(312)
    plt.plot(b, signal-signal, 'k-', b, signal.T-y_recon , 'b-', b, signal.T-y_recon2 , 'r-')
    plt.title('(b) Residual plot')
    a = [abs(np.mean(signal.T-y_recon)), abs(np.mean(signal.T-y_recon2))]
    plt.xlim([0, b[-1]])
    plt.xlabel(['Avg residual: lsqnonneg = '+a[1]+' | NNLSreg = '+a[2]])
    
    plt.subplot(313)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax.plt.semilogx(DValues,s)
    y = plt.ylim
    ytext = y(2)-0.05
    plt.text(1e-3,ytext,'D_{slow}')
    plt.text(4*1e-3,ytext,'D_{inter}')
    plt.text(90*1e-3,ytext,'D_{fast}')
    h = Rectangle([2*1e-3, 0.001, 8*1e-3, ytext+0.05],facecolor=[.9, .9, .9],edgecolor='none')
    plt.uistack(h,'bottom')
    plt.title('(c) Results')
    plt.xlabel('D (mm^2/s) (\cdot 10^{-3})')
    plt.xlim([0.9*1e-3, 200*1e-3])
    plt.ylabel('Amplitude')
    ax2.plt.semilogx(DValues,s2,'r-')
    ax.Color = 'k'
    ax2.Color = 'r'
    return