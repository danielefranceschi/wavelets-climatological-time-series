# global imports
import pandas as pd
from matplotlib import pyplot
import numpy as np
import pycwt as wavelet

# graphic cleanup and initialization function
def initPyPlot(h=8):
    pyplot.close()
    figprops = dict(figsize=(11,h), dpi=96)
    fig = pyplot.figure(**figprops)
    return pyplot.axes()
    
# CWT calculation function
# parameters: t=time array, s=data series
def calculateCWT(t,s,steps=32):
    mother = wavelet.Morlet(6)
    deltaT = t[1] - t[0]
    dj = 1 / steps        # sub-octaves per octaves
    s0 = 2 * deltaT       # Starting scale, here 2 months
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(s, deltaT, dj, s0, -1, mother)
    # Normalized wavelet power spectra
    power = (np.abs(wave)) ** 2
    return power,scales,coi,freqs

# find cycle length
# parameters: power spectrum, scales array, minimum length, maximum length
def findCycleLength(power,scales,startLength,stopLength):
    idxs=next(i for i, v in enumerate(scales) if v>startLength)
    idxe=next(i for i, v in enumerate(scales) if v>stopLength)
    xp=power[idxs:idxe,:]
    # find the maximums' indices
    maxidx=np.argmax(xp,axis=0)+idxs
    # return the periods
    return scales[maxidx]

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
# plot a time series
def plotTimeSeries(time, ser, title, xlabel, ylabel, imageHeight=4, interpolate=False, width=0.5):
    ss=savitzky_golay(ser,63,3) if interpolate else ser
    ax=initPyPlot(imageHeight)
    ax.plot(time,ss,linewidth=width,antialiased=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    pyplot.show()
    
# CWT plot
def plotCWT(time,power,scales,coi,freqs,title,xlabel,ylabel,yTicks=None,steps=512,lowerLimit=0,upperLimitDelta=0):
    zx = initPyPlot()
    
    # cut out very small powers
    LP2=np.log2(power)
    LP2=np.clip(LP2,0,np.max(LP2))
    
    # draw the CWT
    zx.contourf(time, scales, LP2, steps, cmap=pyplot.cm.gist_ncar)
    
    # draw the COI
    coicoi=np.clip(coi,0,coi.max())
    zx.fill_between(time,coicoi,scales.max(),alpha=0.2, color='g', hatch='x')

    # Y-AXIS labels
    if (yTicks):
       yt = yTicks 
    else:
        period=1/freqs
        yt = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))

    zx.set_yscale('log')
    zx.set_yticks(yt)
    zx.set_yticklabels(yt)
    zx.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    
    # exclude some periods from view
    ylim = zx.get_ylim()
    zx.set_ylim(lowerLimit,ylim[1]-upperLimitDelta)
    
    # strings
    zx.set_title(title)
    zx.set_ylabel(ylabel)
    zx.set_xlabel(xlabel)
    # print all
    pyplot.show()