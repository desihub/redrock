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

def _trapz(x, y):
    '''
    trapz integration, about 2x faster than numpy,
    probably because this does no error checking.
    Note that x and y are in opposite order of numpy.trapz(y, x)
    '''
    heights = 0.5*(y[0:-1] + y[1:])
    widths = x[1:] - x[0:-1]
    return np.sum(widths*heights)

def trapz_rebin(x, y, xnew=None, xedges=None):
    '''Rebin y(x) flux density using trapezoidal integration between bin edges
    
    Args:
        x, y : arrays of input y vs. x samples
        xnew : array of new bin centers
        xedges : array of new bin edges
        
        Either `xnew` or `edges` must be provided
        
    Returns:
        array of integrated results with len(results) = len(xedges)-1
        
    Notes:
        y is interpreted as a density, as is the output, e.g.

        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, xedges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])
        
        When xnew is provided, this uses `redrock.rebin.centers2edges()`
        to calculate what you probably mean as the bin edges.
        
    Raises:
        ValueError if edges are outside the range of x
        ValueError if len(x) != len(y)
    '''

    #- Check inputs
    if len(x) != len(y):
        raise ValueError('len(x) {} != len(y) {}'.format(len(x), len(y)))
    if (xedges is None) and (xnew is None):
        raise ValueError('must provide either xnew or xedges')
    if (xedges is not None) and (xnew is not None):
        raise ValueError('must xnew or xedges but not both')

    if xedges is None:
        xedges = centers2edges(xnew)
        
    if xedges[0] < x[0] or x[-1] < xedges[-1]:
        raise ValueError('xedges must be contained within input x range')

    x = np.asarray(x)
    y = np.asarray(y)
    
    #- Pre-calculate y(x) interpolation at bin edges
    yedge = np.interp(xedges, x, y)
    
    #- Trim input xy to just match the range we need
    ii = (xedges[0] < x) & (x < xedges[-1])
    x = x[ii]
    y = y[ii]
    
    #- Loop over pixel, creating temporary regions to integrate that
    #- include the edges and interior y(x) points
    results = np.zeros(len(xedges)-1)
    for i in range(len(xedges)-1):
        ii = np.where((xedges[i] < x) & (x < xedges[i+1]))[0]
        xx = np.zeros(len(ii)+2)
        yy = np.zeros(len(ii)+2)
        xx[0] = xedges[i]
        xx[-1] = xedges[i+1]
        xx[1:-1] = x[ii]
        yy[0] = yedge[i]
        yy[-1] = yedge[i+1]
        yy[1:-1] = y[ii]
        results[i] = _trapz(xx, yy) / (xedges[i+1] - xedges[i])

    return results

#-------------------------------------------------------------------------
#- development tests

if __name__ == '__main__':
    assert np.allclose(centers2edges([1,2,3]), [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(centers2edges([1,3,5]), [0, 2, 4, 6])
    assert np.allclose(centers2edges([1,3,4]), [0, 2, 3.5, 4.5])
    

