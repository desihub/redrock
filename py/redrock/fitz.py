"""
redrock.fitz
============

Functions for fitting minima of chi^2 results.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.constants

from . import constants

from .rebin import rebin_template

from .zscan import calc_zchi2_one, spectral_data, calc_zchi2_batch
from .zscan import calc_negOII_penalty

from .zwarning import ZWarningMask as ZW

from .igm import transmission_Lyman

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

def prior_on_coeffs(n_nbh, deg_legendre, sigma, ncamera):

    """
    Args:
        n_nbh (int): number of dominant archetypes
        deg_legendre (int): number of Legendre polynomials
        sigma (int): prior sigma to be used for archetype fitting
        ncamera (int): number of cameras for given instrument
    Returns:
        2d array to be added while solving for archetype fitting

    """

    nbasis = n_nbh+deg_legendre*ncamera # 3 desi cameras
    prior = np.zeros((nbasis, nbasis), dtype='float64');np.fill_diagonal(prior, 1/(sigma**2))
    for i in range(n_nbh):
        prior[i][i]=0. ## Do not add prior to the archetypes, added only to the Legendre polynomials
    return prior


def fitz(zchi2, redshifts, target, template, nminima=3, archetype=None, use_gpu=False, deg_legendre=None,
         zminfit_npoints=15, per_camera=False, n_nearest=None, prior_sigma=None):
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
        archetype (object, optional): A single Archetype object (of given spectype)
            to use for final fitz choice of best chi2 vs. z minimum.
        use_gpu (bool): use GPU or not
        deg_legendre (int): in archetype mode polynomials upto deg_legendre-1 will be used
        zminfit_npoints (int): number of finer redshift pixels to search for final redshift - default 15
        per_camera: (bool): True if fitting needs to be done in each camera for archetype mode
        n_nearest: (int): number of nearest neighbours to be used in chi2 space (including best archetype)
        prior_sigma (float): prior to add in the final solution matrix: added as 1/(prior_sigma**2) only for per-camera mode

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
    if (zminfit_npoints is None):
        nz = 15
    else:
        nz = zminfit_npoints

    if template.template_type == 'STAR':
        max_velo_diff = constants.max_velo_diff_star
    else:
        max_velo_diff = constants.max_velo_diff

    for imin in find_minima(zchi2):
        if len(results) == nminima:
            break

        #- Skip this minimum if it is within max_velo_diff km/s of a
        # previous one dv is in km/s
        zprev = np.array([tmp['z'] for tmp in results])
        dv = get_dv(z=redshifts[imin],zref=zprev)
        if np.any(np.abs(dv) < max_velo_diff):
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
            T = transmission_Lyman(zz,dwave[k], use_gpu=use_gpu, always_return_array=False,
                                   model=template.igm_model)
            if (T is None):
                #Return value of None means that wavelenght regime
                #does not overlap Lyman transmission - continue here
                continue
            #Vectorize multiplication
            binned[k] *= T[:,:,None]
        if (use_gpu):
            #Use gpu arrays for weights, flux, wflux
            (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, binned, gpuweights, gpuflux, gpuwflux, nz, nbasis,
                                                 solve_matrices_algorithm=template.solve_matrices_algorithm,
                                                 use_gpu=use_gpu)
        else:
            #Use numpy CPU arrays for weights, flux, wflux
            (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, binned, weights, flux, wflux, nz, nbasis,
                                                 solve_matrices_algorithm=template.solve_matrices_algorithm,
                                                 use_gpu=use_gpu)

        #- Penalize chi2 for negative [OII] flux; ad-hoc
        if hasattr(template, 'OIItemplate'):
            zzchi2 += calc_negOII_penalty(template.OIItemplate, zzcoeff)

        #- fit parabola to 3 points around minimum
        i = min(max(np.argmin(zzchi2),1), len(zz)-2)
        zmin, sigma, chi2min, zwarn = minfit(zz[i-1:i+2], zzchi2[i-1:i+2])

        #trans = dict()
        trans = { hs:None for hs, w in dwave.items() } #define trans with keys and None values
        try:
            #Calculate xmin and xmax from template and pass as scalars
            xmin = template.minwave*(1+zmin)
            xmax = template.maxwave*(1+zmin)
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
                T = transmission_Lyman(np.array([zmin]),dwave[k], use_gpu=False, always_return_array=False,
                                       model=template.igm_model)
                trans[k] = T
                if (T is None):
                    #Return value of None means that wavelenght regime
                    #does not overlap Lyman transmission - continue here
                    continue
                #Vectorize multiplication
                binned[k] *= T[:,:,None]
            #Use CPU always with one redshift
            (chi2, coeff) = calc_zchi2_batch(spectra, binned, weights, flux, wflux, 1, nbasis,
                                             solve_matrices_algorithm=template.solve_matrices_algorithm,
                                             use_gpu=False)
            coeff = coeff[0,:]
            pca_coeff = coeff

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
        #- max_velo_diff km/s of a previous one
        zprev = np.array([tmp['z'] for tmp in results])
        dv = get_dv(z=zbest, zref=zprev)
        if np.any(np.abs(dv) < max_velo_diff):
            continue

        if archetype is None:
            results.append(dict(z=zbest, zerr=zerr, zwarn=zwarn,
                chi2=chi2min, zz=zz, zzchi2=zzchi2,
                coeff=coeff, fitmethod=template.method))
        else:
            if prior_sigma is not None:
                if per_camera:
                    ncamera = len(list(dwave.keys())) # number of cameras, for e.g. DESI has three cameras
                else:
                    ncamera = 1
                if n_nearest is None:
                    prior = prior_on_coeffs(1, deg_legendre, prior_sigma, ncamera)
                else:
                    prior = prior_on_coeffs(n_nearest, deg_legendre, prior_sigma, ncamera)
            else:
                prior=None
            chi2min, coeff, fulltype = archetype.get_best_archetype(target,weights,flux,wflux,dwave,zbest, per_camera, n_nearest, trans=trans, use_gpu=use_gpu, prior=prior)
            del trans

            results.append(dict(z=zbest, zerr=zerr, zwarn=zwarn,
                chi2=chi2min, zz=zz, zzchi2=zzchi2,
                coeff=coeff, fulltype=fulltype, fitmethod=archetype.method, pca_method=template.method, pca_coeff=pca_coeff, pca_fulltype=template.full_type))

    #- Sort results by chi2min; detailed fits may have changed order
    ii = np.argsort([tmp['chi2'] for tmp in results])
    results = [results[i] for i in ii]

    if len(results) == 0:
        #- Return blank arrays of 0 minima
        xfloat = np.zeros((0,1), dtype=float)
        xint = np.zeros((0,1), dtype=int)
        xstr = np.zeros((0,1), dtype=str)
        return dict(z=xfloat, zerr=xfloat, zwarn=xint, chi2=xfloat, zz=xfloat, zzchi2=xfloat,
                    coeff=xfloat, fitmethod=xstr, npixels=xint)

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
