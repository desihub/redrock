"""
redrock.zwarning

Mask bit definitions for zwarning

WARNING on the warnings: not all of these are implemented yet.
"""

#- TODO: Consider using something like desispec.maskbits to provide a more
#- convenient wrapper class (probably copy it here; don't make a dependency)
#- That class as-is would bring in a yaml dependency.

class ZWarningMask(object):
    SKY               = 0**2  #- sky fiber
    LITTLE_COVERAGE   = 1**2  #- too little wavelength coverage
    SMALL_DELTA_CHI2  = 2**2  #- chi-squared of best fit is too close to that of second best
    NEGATIVE_MODEL    = 3**2  #- synthetic spectrum is negative (only set for stars and QSOs)
    MANY_OUTLIERS     = 4**2  #- fraction of points more than 5 sigma away from best model is too large (>0.05)
    Z_FITLIMIT        = 5**2  #- chi-squared minimum at edge of the redshift fitting range (Z_ERR set to -1)
    NEGATIVE_EMISSION = 6**2  #- a QSO line exhibits negative emission, triggered only in QSO spectra, if  C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0
    UNPLUGGED         = 7**2  #- the fiber was unplugged, so no spectrum obtained
    BAD_TARGET        = 8**2  #- catastrophically bad targeting data (e.g. ASTROMBAD in CALIB_STATUS)
    NODATA            = 9**2  #- No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)
    BAD_MINFIT        = 10**2 #- Bad parabola fit to the chi2 minimum

