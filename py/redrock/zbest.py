from astropy.table import Table
import numpy as np

def find_zbest(results):
    '''
    Return a table with the best redshift per target
    
    Args:
        results nested dictionary from zfind:
            [targetid][templatetype][z|zchi2|zbest|minchi2|zerr|zwarn]
            
    Returns:
        Table with columns TARGETID, Z, ZERR, ZWARN, TYPE
    '''
    n = len(results)
    zbest = Table()
    zbest['TARGETID'] = np.zeros(n, dtype='i8')
    zbest['Z'] = np.zeros(n, dtype='f4')
    zbest['ZERR'] = np.zeros(n, dtype='f4')
    zbest['ZWARN'] = np.zeros(n, dtype='i8')
    zbest['SPECTYPE'] = np.zeros(n, dtype='S10')
    zbest['DELTACHI2'] = np.zeros(n, dtype='f4')

    for i, targetid in enumerate(results):
        chi2min = 1e120
        for ttype in results[targetid]:
            rx = results[targetid][ttype]['minima'][0]  #- best fit from each target class                    
            if rx['chi2'] < chi2min:
                ### deltachi2 = min(rx['deltachi2'], chi2min-rx['minchi2'])
                chi2min = rx['chi2']
                z = rx['z']
                zerr = rx['zerr']
                zwarn = rx['zwarn']
                type_ = ttype
    
        zbest['TARGETID'][i] = targetid
        zbest['Z'][i] = z
        zbest['ZERR'][i] = zerr
        zbest['ZWARN'][i] = zwarn
        zbest['SPECTYPE'][i] = type_
        ### zbest['DELTACHI2'][i] = deltachi2
    
    return zbest
