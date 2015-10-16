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
    
    #- Pre-calculate y(x) interpolation at bin edges
    yedge = np.interp(edges, x, y)
    
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
        yy[0] = yedge[i]
        yy[-1] = yedge[i+1]
        yy[1:-1] = y[ii]
        results[i] = _trapz(xx, yy) / (edges[i+1] - edges[i])

    return results

#-------------------------------------------------------------------------
#- development tests

if __name__ == '__main__':
    assert np.allclose(centers2edges([1,2,3]), [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(centers2edges([1,3,5]), [0, 2, 4, 6])
    assert np.allclose(centers2edges([1,3,4]), [0, 2, 3.5, 4.5])
    

