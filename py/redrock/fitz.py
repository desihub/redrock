"""
redrock.fitz
============

Functions for fitting minima of chi^2 results.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.constants
import scipy.special

from . import constants

from .rebin import rebin_template

from .zscan import calc_zchi2_one, spectral_data, calc_zchi2_batch

from .zwarning import ZWarningMask as ZW

from .utils import transmission_Lyman

def get_dv(z, zref):
    """Returns velocity difference in km/s for two redshifts

    Args:
        z (float): redshift for comparison.
        zref (float): reference redshift.

    Returns:
        (float): the velocity difference.

    """

    c = (scipy.constants.speed_of_light/1000.) #- km/s
    dv = c * (z - zref) / (1.0 + zref)

    return dv


def find_minima(x):
    """Return indices of local minima of x, including edges.

    The indices are sorted small to large.

    Note:
        this is somewhat conservative in the case of repeated values:
        find_minima([1,1,1,2,2,2]) -> [0,1,2,4,5]

    Args:
        x (array-like): The data array.

    Returns:
        (array): The indices.

    """
    x = np.asarray(x)
    ii = np.where(np.r_[True, x[1:]<=x[:-1]] & np.r_[x[:-1]<=x[1:], True])[0]

    jj = np.argsort(x[ii])

    return ii[jj]


def minfit(x, y):
    """Fits y = y0 + ((x-x0)/xerr)**2

    See redrock.zwarning.ZWarningMask.BAD_MINFIT for zwarn failure flags

    Args:
        x (array): x values.
        y (array): y values.

    Returns:
        (tuple):  (x0, xerr, y0, zwarn) where zwarn=0 is good fit.

    """
    if len(x) < 3:
        return (-1,-1,-1,ZW.BAD_MINFIT)

    try:
        #- y = a x^2 + b x + c
        a,b,c = np.polyfit(x,y,2)
    except np.linalg.LinAlgError:
        return (-1,-1,-1,ZW.BAD_MINFIT)

    if a == 0.0:
        return (-1,-1,-1,ZW.BAD_MINFIT)

    #- recast as y = y0 + ((x-x0)/xerr)^2
    x0 = -b / (2*a)
    y0 = -(b**2) / (4*a) + c

    zwarn = 0
    if (x0 <= np.min(x)) or (np.max(x) <= x0):
        zwarn |= ZW.BAD_MINFIT
    if (y0<=0.):
        zwarn |= ZW.BAD_MINFIT

    if a > 0.0:
        xerr = 1 / np.sqrt(a)
    else:
        xerr = 1 / np.sqrt(-a)
        zwarn |= ZW.BAD_MINFIT

    return (x0, xerr, y0, zwarn)



def legendre_calculate(nleg, dwave):
    wave = np.concatenate([ w for w in dwave.values() ])
    wmin = wave.min()
    wmax = wave.max()
    legendre = { hs:np.array([scipy.special.legendre(i)( (w-wmin)/(wmax-wmin)*2.) for i in range(nleg)]) for hs, w in dwave.items() }

    return legendre

def fitz(zchi2, redshifts, target, template, nminima=3, archetype=None, use_gpu=False, deg_legendre=None, nz=15, per_camera=False, n_nearest=None):
    """Refines redshift measurement around up to nminima minima.

    TODO:
        if there are fewer than nminima minima, consider padding.

    Args:
        zchi2 (array): chi^2 values for each redshift.
        redshifts (array): the redshift values.
        target (Target): the target for this fit which includes a list
            of Spectrum objects at different wavelength grids.
        template (Template): the template for this fit.
        nminima (int): the number of minima to consider.
        use_gpu (bool): use GPU or not
        deg_legendre (int): in archetype mode polynomials upto deg_legendre-1 will be used
        nz (int): number of finer redshift pixels to search for final redshift - default 15
        per_camera: (bool): True if fitting needs to be done in each camera for archetype mode
        n_nearest: (int): number of nearest neighbours to be used in chi2 space (including best archetype) 

    Returns:
        Table: the fit parameters for the minima.

    """
    assert len(zchi2) == len(redshifts)
    #Import cupy locally if using GPU
    if (use_gpu):
        import cupy as cp

    nbasis = template.nbasis
    spectra = target.spectra

    # Build dictionary of wavelength grids
    dwave = { s.wavehash:s.wave for s in spectra }

    (weights, flux, wflux) = spectral_data(spectra)
    
    if (use_gpu):
        #Get CuPy arrays of weights, flux, wflux
        #These are created on the first call of gpu_spectral_data() for a
        #target and stored.  They are retrieved on subsequent calls.
        (gpuweights, gpuflux, gpuwflux) = target.gpu_spectral_data()
        # Build dictionaries of wavelength bin edges, min/max, and centers
        gpuedges = { s.wavehash:(s.gpuedges, s.minedge, s.maxedge) for s in spectra }
        gpudwave = { s.wavehash:s.gpuwave for s in spectra }

    if not archetype is None:
        #legendre = legendre_calculate(deg_legendre, dwave=dwave)
        legendre = target.legendre(deg_legendre)

    results = list()
    #Moved default nz to arg list
    if (nz is None):
        nz = 15

    for imin in find_minima(zchi2):
        if len(results) == nminima:
            break

        #- Skip this minimum if it is within constants.max_velo_diff km/s of a
        # previous one dv is in km/s
        zprev = np.array([tmp['z'] for tmp in results])
        dv = get_dv(z=redshifts[imin],zref=zprev)
        if np.any(np.abs(dv) < constants.max_velo_diff):
            continue

        #- Sample more finely around the minimum
        ilo = max(0, imin-1)
        ihi = min(imin+1, len(zchi2)-1)
        zz = np.linspace(redshifts[ilo], redshifts[ihi], nz)
        if (use_gpu):
            #Create a redshift grid on the GPU as well
            gpuzz = cp.asarray(zz)

        zzchi2 = np.zeros(nz, dtype=np.float64)
        zzcoeff = np.zeros((nz, nbasis), dtype=np.float64)

        #Calculate xmin and xmax from template and zz array on CPU and
        #pass as scalars
        xmin = template.minwave*(1+zz.max())
        xmax = template.maxwave*(1+zz.min())

        #Use batch mode for rebin_template, transmission_Lyman, and calc_zchi2
        if (use_gpu):
            #Use gpuedges already calculated and on GPU
            binned = rebin_template(template, gpuzz, dedges=gpuedges, use_gpu=use_gpu, xmin=xmin, xmax=xmax)
        else:
            #Use numpy CPU arrays
            binned = rebin_template(template, zz, dwave, use_gpu=use_gpu, xmin=xmin, xmax=xmax)
        # Correct spectra for Lyman-series
        for k in list(dwave.keys()):
            #New algorithm accepts all z as an array and returns T, a 2-d
            # matrix (nz, nlambda) as a cupy or numpy array
            T = transmission_Lyman(zz,dwave[k], use_gpu=use_gpu)
            if (T is None):
                #Return value of None means that wavelenght regime
                #does not overlap Lyman transmission - continue here
                continue
            #Vectorize multiplication
            binned[k] *= T[:,:,None]
        if (use_gpu):
            #Use gpu arrays for weights, flux, wflux
            (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, binned, gpuweights, gpuflux, gpuwflux, nz, nbasis, use_gpu=use_gpu)
        else:
            #Use numpy CPU arrays for weights, flux, wflux 
            (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, binned, weights, flux, wflux, nz, nbasis, use_gpu=use_gpu)

        #- fit parabola to 3 points around minimum
        i = min(max(np.argmin(zzchi2),1), len(zz)-2)
        zmin, sigma, chi2min, zwarn = minfit(zz[i-1:i+2], zzchi2[i-1:i+2])

        trans = dict()
        try:
            #Calculate xmin and xmax from template and pass as scalars 
            xmin = template.minwave*(1+zmin)
            ximax = template.maxwave*(1+zmin)
            if (use_gpu):
                #Use gpuedges already calculated and on GPU
                binned = rebin_template(template, cp.array([zmin]), dedges=gpuedges, use_gpu=use_gpu, xmin=xmin, xmax=xmax)
            else:
                binned = rebin_template(template, np.array([zmin]), dwave, use_gpu=use_gpu, xmin=xmin, xmax=xmax)
            for k in list(dwave.keys()):
                if (use_gpu):
                    #Copy binned[k] back to CPU to perform next steps on CPU
                    #because faster with only 1 redshift
                    binned[k] = binned[k].get()
                #Use CPU always
                T = transmission_Lyman(np.array([zmin]),dwave[k], use_gpu=False)
                trans[k] = T
                if (T is None):
                    #Return value of None means that wavelenght regime
                    #does not overlap Lyman transmission - continue here
                    continue
                #Vectorize multiplication
                binned[k] *= T[:,:,None]
            #Use CPU always with one redshift
            (chi2, coeff) = calc_zchi2_batch(spectra, binned, weights, flux, wflux, 1, nbasis, use_gpu=False)
            coeff = coeff[0,:]
        except ValueError as err:
            if zmin<redshifts[0] or redshifts[-1]<zmin:
                #- beyond redshift range can be invalid for template
                coeff = np.zeros(template.nbasis)
                zwarn |= ZW.Z_FITLIMIT
                zwarn |= ZW.BAD_MINFIT
            else:
                #- Unknown problem; re-raise error
                raise err

        zbest = zmin
        zerr = sigma

        #- parabola minimum outside fit range; replace with min of scan
        if zbest < zz[0] or zbest > zz[-1]:
            zwarn |= ZW.BAD_MINFIT
            imin = np.where(zzchi2 == np.min(zzchi2))[0][0]
            zbest = zz[imin]
            chi2min = zzchi2[imin]

        #- Initial minimum or best fit too close to edge of redshift range
        if zbest < redshifts[1] or zbest > redshifts[-2]:
            zwarn |= ZW.Z_FITLIMIT
        if zmin < redshifts[1] or zmin > redshifts[-2]:
            zwarn |= ZW.Z_FITLIMIT

        #- Skip this better defined minimum if it is within
        #- constants.max_velo_diff km/s of a previous one
        zprev = np.array([tmp['z'] for tmp in results])
        dv = get_dv(z=zbest, zref=zprev)
        if np.any(np.abs(dv) < constants.max_velo_diff):
            continue

        if archetype is None:
            results.append(dict(z=zbest, zerr=zerr, zwarn=zwarn,
                chi2=chi2min, zz=zz, zzchi2=zzchi2,
                coeff=coeff))
        else:
            chi2min, coeff, fulltype = archetype.get_best_archetype(target,weights,flux,wflux,dwave,zbest, per_camera, n_nearest, trans=trans, use_gpu=use_gpu)
            del trans

            results.append(dict(z=zbest, zerr=zerr, zwarn=zwarn,
                chi2=chi2min, zz=zz, zzchi2=zzchi2,
                coeff=coeff, fulltype=fulltype))

    #- Sort results by chi2min; detailed fits may have changed order
    ii = np.argsort([tmp['chi2'] for tmp in results])
    results = [results[i] for i in ii]

    assert len(results) > 0
    #- Convert list of dicts -> Table
    #from astropy.table import Table
    #results = Table(results)

    # astropy Table is really slow, Finalizing is 8x faster
    # using dict of np arrays

    #Move npixels summation here from zfind.py
    for i in range(len(results)):
        results[i]['npixels'] = 0
        for s in spectra:
            results[i]['npixels'] += (s.ivar>0.).sum()
    #Create dict here.  np.vstack essentially does the same thing
    #as putting in an astropy Table -> results is converted from
    #a list of dicts of scalars and 1d arrays to a single dict
    #with 1d and 2d np arrays.
    tmp = dict()
    for k in results[0].keys():
        tmp[k] = list()
        for i in range(len(results)):
            tmp[k].append(results[i][k])
        tmp[k] = np.vstack(tmp[k])
    results = tmp

    return results
