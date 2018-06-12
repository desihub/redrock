"""
redrock.rebin
=============

Tools for binning data.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from ._rebin import _trapz_rebin

from .utils import mp_array

from desispec.interpolation import resample_flux


def centers2edges(centers):
    """Convert bin centers to bin edges, guessing at what you probably meant

    Args:
        centers (array): bin centers,

    Returns:
        array: bin edges, lenth = len(centers) + 1

    """
    centers = np.asarray(centers)
    edges = np.zeros(len(centers)+1)
    #- Interior edges are just points half way between bin centers
    edges[1:-1] = (centers[0:-1] + centers[1:]) / 2.0
    #- edge edges are extrapolation of interior bin sizes
    edges[0] = centers[0] - (centers[1]-edges[1])
    edges[-1] = centers[-1] + (centers[-1]-edges[-2])

    return edges


def trapz_rebin(x, y, xnew=None, edges=None):
    """Rebin y(x) flux density using trapezoidal integration between bin edges

    Notes:
        y is interpreted as a density, as is the output, e.g.

        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, edges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])

    Args:
        x (array): input x values.
        y (array): input y values.
        edges (array): (optional) new bin edges.

    Returns:
        array: integrated results with len(results) = len(edges)-1

    Raises:
        ValueError: if edges are outside the range of x or if len(x) != len(y)

    """
    if edges is None:
        edges = centers2edges(xnew)
    else:
        edges = np.asarray(edges)

    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be within input x range')

    result = np.zeros(len(edges)-1, dtype=np.float64)

    _trapz_rebin(x, y, edges, result)

    return result


def rebin_template(template, z, dwave):
    """Rebin a template to a set of wavelengths.

    Given a template and a single redshift, rebin the template to a set of
    wavelength arrays.

    Args:
        template (Template): the template object
        z (float): the redshift
        dwave (dict): the keys are the "wavehash" and the values
            are a 1D array containing the wavelength grid.

    Returns:
        dict:  The rebinned template for every basis function and wavelength
            grid in dwave.

    """
    nbasis = template.flux.shape[0]  #- number of template basis vectors
    result = { hs:np.array([ resample_flux(wave, (1.+z)*template.wave, template.flux[b]) for b in range(nbasis) ]).transpose() for hs, wave in dwave.items() }
    return result
