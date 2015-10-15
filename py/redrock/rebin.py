import numpy as np

def centers2edges(centers):
    '''convert bin centers to bin edges, guessing at what you probably meant
    
    Args:
        centers : array of bin centers
    
    Returns:
        edges : array of bin edges, lenth = len(centers) + 1
    '''
    centers = np.asarray(centers)
    edges = np.zeros(len(centers)+1)
    #- Interior edges are just points half way between bin centers
    edges[1:-1] = (centers[0:-1] + centers[1:]) / 2.0
    #- edge edges are extrapolation of interior bin sizes
    edges[0] = centers[0] - (centers[1]-edges[1])
    edges[-1] = centers[-1] + (centers[-1]-edges[-2])
    
    return edges

def trapz_rebin(edges, x, y):
    '''Rebin y(x) flux density using trapezoidal integration between bin edges
    
    Args:
        edges : array of new bin edges
        x, y : arrays of input y vs. x samples
        
    Returns:
        array of integrated results with len(results) = len(edges)-1
        
    Notes:
        y is interpreted as a density, as is the output, e.g.

        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin([0,2,4,6,8])  #- density preserved as 1, not 2
        array([ 1.,  1.,  1.,  1.])
        
    Raises:
        ValueError if edges are outside the range of x
        ValueError if len(x) != len(y)
    '''

    #- Check inputs
    if len(x) != len(y):
        raise ValueError('len(x) {} != len(y) {}'.format(len(x), len(y)))
    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be contained within input x range')

    x = np.asarray(x)
    y = np.asarray(y)
    
    #- Loop over pixel, creating temporary regions to integrate that
    #- include the edges and interior y(x) points
    results = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        ii = np.where((edges[i] < x) & (x < edges[i+1]))[0]
        xx = np.zeros(len(ii)+2)
        yy = np.zeros(len(ii)+2)
        xx[0] = edges[i]
        xx[-1] = edges[i+1]
        xx[1:-1] = x[ii]
        yy[0] = np.interp(edges[i], x, y)
        yy[-1] = np.interp(edges[i+1], x, y)
        yy[1:-1] = y[ii]
        results[i] = np.trapz(yy, xx) / (edges[i+1] - edges[i])
        
    return results
    
def nyquist_rebin(edges, x, y):
    '''
    rebin a Nyquist sampled spectrum y(x) along new bin edges
    
    NOTE: THIS DOESN'T WORK UNLESS YOUR INPUT REALLY IS BAND LIMITED AND
    YOU ARE *EXACTLY* NYQUIST SAMPLED.
    
    i.e. don't use this, but i'm not quite willing to delete the code yet
    '''

    if len(x) != len(y):
        raise ValueError('len(x) {} != len(y) {}'.format(len(x), len(y)))
    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be contained within input x range')

    #- Use FFT to get representation of Nyquist sampled spectrum
    #- y(x) = sum_i a_i cos(x*2pi*i/n) + b_i sin(x*2pi*i/n)
    n = len(x)
    f = fftpack.fft(y - np.mean(y))
    a = np.real(f)/n
    b = np.imag(f)/n

    #- Should be able to get the same thing back
    twopi = 2*np.pi
    newy = np.zeros_like(y) + np.mean(y)
    ex = np.arange(len(x))
    for i in range(n):
        xx = ex*i*twopi/n
        newy += a[i]*np.cos(xx) - b[i]*np.sin(xx)

    assert np.allclose(y, newy)

    #- convert edges to units of input bins
    ex = np.interp(edges, x, np.arange(len(x)))

    #- Integral over bins
    z = np.zeros_like(edges) + np.mean(y)*ex
    for i in range(1,n):
        xx = ex*i*twopi/n
        z += (a[i]*np.sin(xx) + b[i]*np.cos(xx)) * (n/twopi/i)
        
    nbin = len(edges)-1
    zz = (z[1:1+nbin] - z[0:nbin]) / (edges[1:1+nbin] - edges[0:nbin])
    return zz    

#-------------------------------------------------------------------------
#- development tests

if __name__ == '__main__':
    assert np.allclose(centers2edges([1,2,3]), [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(centers2edges([1,3,5]), [0, 2, 4, 6])
    assert np.allclose(centers2edges([1,3,4]), [0, 2, 3.5, 4.5])
    

