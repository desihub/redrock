import numpy as np
import redrock.zscan

def pickz(zchi2, redshifts, spectra, templates):
    '''Identifies which template has the best redshift and refines the z answer
    zchi2[num_templates, num_redshifts]
    '''
    itemplate = np.argmin(np.min(zchi2, axis=1))
    iz = np.argmin(zchi2[itemplate])
    zmin = redshifts[iz]
    print "Best chi2 for template {} at z={:0.5f}".format(itemplate, zmin)
    
    dz = 0.005
    zz = np.linspace(zmin-dz, zmin+dz, 51)
    zchi2_new = redrock.zscan.calc_zchi2_template(zz, spectra, templates[itemplate])
    
    inew = np.argmin(zchi2_new)
    znew = zz[inew]
    print "New zmin at {:0.5f}".format(znew)
    
    #- Fit parabola
    ii = np.where(zchi2_new <= zchi2_new[inew]+9)[0]
    p = np.polyfit(zz[ii], zchi2_new[ii], 2)
        
    #- Evaluate zbest at minimum of fitted parabola
    a, b, c = p
    zbest = -b/(2*a)
    chi2min = c - b**2/(4*a)
    zp = (-b + np.sqrt(b**2 - 4*a*(c-1-chi2min))) / (2*a)
    zerr = zp-zbest

    return zbest, zerr

