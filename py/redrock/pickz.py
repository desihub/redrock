import numpy as np
import redrock.zscan

def pickz(zchi2, redshifts, spectra, template):
    '''Refines redshift measurement
    '''
    zmin = redshifts[np.argmin(zchi2)]
    dz = 0.001
    zz = np.linspace(zmin-dz, zmin+dz, 21)
    zzchi2 = redrock.zscan.calc_zchi2(zz, spectra, template)
    chi2min = np.min(zzchi2)

    zmin = zz[np.argmin(zzchi2)]

    jj = (zzchi2 < np.min(zzchi2)+50)
    p = np.polyfit(zz[jj], zzchi2[jj], 2)

    a, b, c = p
    zbest = -b/(2*a)
    chi2min = c - b**2/(4*a)
    zp = (-b + np.sqrt(b**2 - 4*a*(c-1-chi2min))) / (2*a)
    zerr = zp-zbest

    return zbest, zerr

