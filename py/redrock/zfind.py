import numpy as np

import redrock.zscan
import redrock.pickz

def zfind(targets, templates):
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
    
