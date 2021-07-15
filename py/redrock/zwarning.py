"""
redrock.zwarning
================

Mask bit definitions for zwarning.

WARNING on the warnings: not all of these are implemented yet.
"""

#- TODO: Consider using something like desispec.maskbits to provide a more
#- convenient wrapper class (probably copy it here; don't make a dependency)
#- That class as-is would bring in a yaml dependency.

class ZWarningMask(object):
    SKY               = 2**0  #- sky fiber
    LITTLE_COVERAGE   = 2**1  #- too little wavelength coverage
    SMALL_DELTA_CHI2  = 2**2  #- chi-squared of best fit is too close to that of second best
    NEGATIVE_MODEL    = 2**3  #- synthetic spectrum is negative
    MANY_OUTLIERS     = 2**4  #- fraction of points more than 5 sigma away from best model is too large (>0.05)
    Z_FITLIMIT        = 2**5  #- chi-squared minimum at edge of the redshift fitting range
    NEGATIVE_EMISSION = 2**6  #- a QSO line exhibits negative emission, triggered only in QSO spectra, if  C_IV, C_III, Mg_II, H_beta, or H_alpha has LINEAREA + 3 * LINEAREA_ERR < 0
    UNPLUGGED         = 2**7  #- the fiber was unplugged/broken, so no spectrum obtained
    BAD_TARGET        = 2**8  #- catastrophically bad targeting data
    NODATA            = 2**9  #- No data for this fiber, e.g. because spectrograph was broken during this exposure (ivar=0 for all pixels)
    BAD_MINFIT        = 2**10 #- Bad parabola fit to the chi2 minimum
    POORDATA          = 2**11 #- Poor input data quality but try fitting anyway

    #- The following bits are reserved for experiment-specific post-redrock
    #- afterburner updates to ZWARN; redrock commits to *not* setting these bits
    RESERVED16        = 2**16
    RESERVED17        = 2**17
    RESERVED18        = 2**18
    RESERVED19        = 2**19
    RESERVED20        = 2**20
    RESERVED21        = 2**21
    RESERVED22        = 2**22
    RESERVED23        = 2**23

    @classmethod
    def flags(cls):
        flagmask = list()
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and key.isupper():
                flagmask.append((key, value))

        import numpy as np
        isort = np.argsort([x[1] for x in flagmask])
        flagmask = [flagmask[i] for i in isort]
        return flagmask
