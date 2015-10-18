from __future__ import division, print_function

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
def trapz_rebin(x, y, xnew=None, edges=None):
    if edges is None:
        edges = centers2edges(xnew)

    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be within input x range')

    return _trapz_rebin(x, y, edges)

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
    edges = np.asarray(edges)
    
    nbin = len(edges)-1
    nx = len(x)
    results = np.zeros(nbin)
    i = 0  #- index counter for output
    j = 0  #- index counter for inputs
    
    #- Find edge and include first trapezoid
    # print('---')
    # print(x)
    # print('edges =', repr(edges))
    while i < nbin:
        #- Seek next sample beyond bin edge
        while x[j] <= edges[i]:
            j += 1

        #- What is the y value where the interpolation crossed the edge?
        yedge = y[j-1] + (edges[i]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])

        #- Is this sample inside this bin?
        if x[j] < edges[i+1]:
            area = 0.5 * (y[j] + yedge) * (x[j] - edges[i])
            # print('edge_dn ', i, j, area)
            results[i] += area

            #- Continue with interior bins
            while x[j+1] < edges[i+1]:
                j += 1
                area = 0.5 * (y[j] + y[j-1]) * (x[j] - x[j-1])
                # print('interior', i, j, area)
                results[i] += area
            
            #- Next sample will be outside this bin; handle upper edge
            yedge = y[j] + (edges[i+1]-x[j]) * (y[j+1]-y[j]) / (x[j+1]-x[j])
            area = 0.5 * (yedge + y[j]) * (edges[i+1] - x[j])
            # print('edge_up ', i, j, area)
            results[i] += area

        #- Otherwise the samples span over this bin
        else:
            ylo = y[j] + (edges[i]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            yhi = y[j] + (edges[i+1]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            area = 0.5 * (ylo+yhi) * (edges[i+1]-edges[i])
            # print('step over', i, j, area)
            results[i] += area

        i += 1

    return results / (edges[1:] - edges[0:-1])
    

