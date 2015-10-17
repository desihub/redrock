import numpy as np
import numba

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

#- numba JIT compiler doesn't seem to like keyword args, so wrap it
def trapz_rebin(x, y, xnew=None, xedges=None):
    if xedges is None:
        xedges = centers2edges(xnew)

    return _trapz_rebin(x, y, xedges)

@numba.jit
def _trapz_rebin(x, y, edges):
    '''Rebin y(x) flux density using trapezoidal integration between bin edges
    
    Args:
        x, y : arrays of input y vs. x samples
        edges : array of new bin edges
                
    Returns:
        array of integrated results with len(results) = len(edges)-1
        
    Notes:
        y is interpreted as a density, as is the output, e.g.

        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, edges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])
                
    Raises:
        ValueError if edges are outside the range of x
        ValueError if len(x) != len(y)
    '''
    assert x[0] < edges[0]
    assert x[-1] > edges[-1]
    
    edges = np.asarray(edges)
    
    n = len(edges)-1
    results = np.zeros(n)
    i = 0  #- index counter for output
    j = 0  #- index counter for inputs
    
    #- Find edge and include first trapezoid
    
    while i < n:
        while x[j] <= edges[i]:
            j += 1

        yedge = y[j-1] + (edges[i]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])
        if i > 0:
            blat = 0.5 * (yedge + y[j-1]) * (edges[i] - x[j-1])
            results[i-1] += blat

        blat = 0.5 * (y[j] + yedge) * (x[j] - edges[i])
        results[i] += blat

        #- Continue with interior bins
        j += 1
        while x[j] < edges[i+1]:
            blat = 0.5 * (y[j] + y[j-1]) * (x[j] - x[j-1])
            results[i] += blat
            j += 1
            
        #- Get last edge trap
        if i == n-1:
            yedge = y[j-1] + (edges[i+1]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])
            blat = 0.5 * (yedge + y[j-1]) * (edges[i+1] - x[j-1])
            results[i] += blat

        i += 1

    return results / (edges[1:] - edges[0:-1])
    
#-------------------------------------------------------------------------
#- development tests

if __name__ == '__main__':
    assert np.allclose(centers2edges([1,2,3]), [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(centers2edges([1,3,5]), [0, 2, 4, 6])
    assert np.allclose(centers2edges([1,3,4]), [0, 2, 3.5, 4.5])
    

