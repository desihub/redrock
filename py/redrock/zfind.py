import numpy as np

import redrock.zscan
import redrock.pickz

def zfind(targets, templates):
    '''
    Given a list of targets and a list of templates, find redshifts
    
    Args:
        targets : list of (targetid, spectra), where spectra are a list of
            dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        templates: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux[i,wave] : template basis vectors of flux densities
        
    Returns nested dictionary results[targetid][templatetype] with keys
        - z: array of redshifts scanned
        - zchi2: array of chi2 fit at each z
        - zbest: best fit redshift (finer resolution fit around zchi2 minimum)
        - minchi2: chi2 at zbest
        - zerr: uncertainty on zbest
        - zwarn: 0=good, non-0 is a warning flag    
    '''
    redshifts = dict(
        ELG  = 10**np.arange(np.log10(0.1), np.log10(1.8), 4e-4),
        LRG  = 10**np.arange(np.log10(0.1), np.log10(1.3), 4e-4),
        STAR = np.arange(-0.001, 0.00101, 0.0001),
        #'QSO'...
    )

    #- Try each template on the spectra for each target
    results = dict()    
    for targetid, spectra in targets:
        results[targetid] = dict()
        for t in templates:
            ttype = t['type']+t['subtype']
            if t['type'] == 'GALAXY':  #- clumsy...
                zz = redshifts[t['subtype']]   #- ELG or LRG
            else:
                zz = redshifts[t['type']]
            
            zchi2 = redrock.zscan.calc_zchi2(zz, spectra, t)
        
            zbest, zerr, zwarn, minchi2 = redrock.pickz.pickz(zchi2, zz, spectra, t)
        
            results[targetid][ttype] = dict(
                z=zz, zchi2=zchi2, zbest=zbest, zerr=zerr, zwarn=zwarn, minchi2=minchi2
            )
        
            print('{} {:6s} {:4s} {:.6f} {:.6f} {:6d} {:.2f}'.format(
                targetid, t['type'], t['subtype'], zbest, zerr, zwarn, minchi2))
                
    return results
    
