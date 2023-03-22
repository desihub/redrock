## Best Archetype redshift estimateion based on Nearest neighbour approach on synthetic spectra approach

import numpy as np
from astropy.io import fits
from scipy.spatial.distance import euclidean
from numba import jit

def read_fits_data(filename, nhdu):
    """
     Reads hdu data from a fits file for given hdu and returns fits data    
    """
    hdu = fits.open(filename)
    data = hdu[nhdu].data
    hdu.close()
    return data


def synthetic_galaxy_data_read(gals):

    """ Returns fluxes and physical parameters (dictionary) 
        for synthetic galaxy spectra generated with desisim
        by Abhijeet Anand, LBL, email: AbhijeetAnand@lbl.gov
    """
    all_gal_data = {}
    file_gals = '/global/cfs/cdirs/desi/users/abhijeet/synthetic_spectra/%s/%s_spectral_flux_rest_frame.fits'%(gals, gals)
    temp_gal_data = read_fits_data(file_gals, nhdu=2)
    all_gal_data['FLUX'] = fits.open(file_gals)[1].data
    for key in temp_gal_data.dtype.names:
        all_gal_data[key] = temp_gal_data[key].data
    return all_gal_data

@jit(nopython=True)
def cartesian_dist(x_ref, Y, n_nearest):

    """
        Calculates 3d euclidean distance between reference and target array points and returns
        n-nearest distances and indices of Y (two arrays)

    Input:
        x_ref: [list or 1d array], reference 3d coordinate
        Y: [2d array], each row should be 3d coordinate of one object
        n_nearest: [int]: return n_nearest objects (theri indices)
    """
    if isinstance(x_ref, list):
        x_ref = np.array(x_ref)

    dist =  np.array([euclidean(x_ref, c) for c in Y])
    inds = np.argsort(dist)
    return dist, inds[0:n_nearest]

def return_N_nearest_archetypes_from_synthetic_spectra(arch_id, archetype_data, gal_data, n_nbh, ret_wave=True):

    """ Returns the fluxes and rest-frame wavelength (same as archetypes) of N-nearest neighbours for a given Best 
        archetypes after estimating the distance between it and all archetypes

    Input: 
        arch_id: [int]; best archetype id found by redrock in archetype mode
        archetype_data: [dict]; dictionary containing physical parameters for archetypes
        must contain:  ['LOGMSTAR [M_sol]', 'LOGSSFR [yr^-1]', 'AV_ISM [mag]'] --> 3d coordinate...

        gal_data: [dict]; full synthetic galaxy data dictionary, 
        must contain:['LOGMSTAR [M_sol]', 'LOGSSFR [yr^-1]', 'AV_ISM [mag]', 'FLUX'] ....

        n_nbh: [int], number of nearest neighbours you want to return
        ret_wave: [bool], returns the wavelength array if True (default True)
    """
    param_keys = ['LOGMSTAR [M_sol]', 'LOGSSFR [yr^-1]', 'AV_ISM [mag]']
    Y =  np.array([gal_data[key] for key in param_key]).T  # parameters for synthetic galaxies (3d coordinates)
    Xref = np.array(list(archetype_data[[param_keys]][arch_id])) # parameter for best fit redrock archetypes
    dist, ii = cartesian_dist(x_ref=Xref, Y=Y, n_nbh=n_nbh)
    arch_flux = gal_data['FLUX'][ii]
    if ret_wave:
        lam0 = 1228.0701754386 #starting rest frame wavelength (same as PCA templates)
        npix = 19545 # number of wavelength pixels same as archetypes (5 times lower than PCA)
        step_size = 0.5 # pixel size in Angstrom (5 times lower resolution than PCA templates)
        temp_wave = lam0 + np.arange(npix)*step_size # Galaxy archetype wavelengths
        return arch_flux, temp_wave, ii
    else:
        return arch_flux, ii


def return_galaxy_archetype_properties(archetypes):

    """ Returns a dictionary containing the physical properties of archetype galaxies
        generated from desisim by Abhijeet Anand, LBL, email: AbhijeetAnand@lbl.gov
        Returns: ['LOGMSTAR [M_sol]', 'LOGSSFR [yr^-1]', 'AV_ISM [mag]']
    """

    if os.path.isfile(archetypes):
        filename = archetypes
    if os.path.isdir(archetypes):
        filename = archetypes+'rrarchetype-galaxy.fits'
    data= read_fits_data(filename, nhdu=2)
    params = {}
    for key in data.dtype.names():
        params[key] = data[key]
    return params

def params_for_all_galaxies():
    """
        Returns dictionaries for all galaxies (ELG, LRG, BGS)
        access each dictionary as fin_dict[subtype]
    """
    synthetic_galaxies = {}
    for galaxies in ['ELG', 'LRG', 'BGS']:
        synthetic_galaxies[galaxies] = synthetic_galaxy_data_read(galaxies)
    return synthetic_galaxies



