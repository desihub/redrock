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
    zbest['TYPE'] = np.zeros(n, dtype='S10')

    for i, targetid in enumerate(results):
        chi2min = 1e120
        for ttype in results[targetid]:
            rx = results[targetid][ttype]
            if rx['minchi2'] < chi2min:
                chi2min = rx['minchi2']
                z = rx['zbest']
                zerr = rx['zerr']
                zwarn = rx['zwarn']
                type_ = ttype
    
            zbest['TARGETID'][i] = targetid
            zbest['Z'][i] = z
            zbest['ZERR'][i] = zerr
            zbest['ZWARN'][i] = zwarn
            zbest['TYPE'][i] = type_
    
    return zbest
