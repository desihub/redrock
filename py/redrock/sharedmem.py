'''
Utilities for working with multiprocessing shared memory arrays

Copy a numpy array into a shared memory tuple::

    raw = sharedmem.fromarray(np.arange(1000))

`raw` can be passed to a separate multiprocessing process with minimal
pickling and memory overhead, and then converted back into a numpy ndarray::

    x = sharedmem.toarray(raw)

Writing to x will change the shared buffer for all processes, but this does
not provide any locking mechanism.  In general this is intended for large
read-only inputs.
'''

import numpy as np
import multiprocessing as mp

def fromarray(x):
    '''
    Copies array `x` into a shared memory buffer

    Returns `raw` tuple of information for `y = toarray(shmem)`

    Note: this copies the contents of `x`; it does *not* convert `x` itself
    into a shared memory array
    '''
    x = np.asarray(x)
    rawbuf = mp.RawArray(x.dtype.char, x.size)
    a = np.frombuffer(rawbuf, dtype=x.dtype).reshape(x.shape)
    a[:] = x
    return (rawbuf, x.dtype, x.shape)

def toarray(raw):
    '''
    Generate a numpy.ndarray with info from sharedmem.fromarray(x)
    '''
    rawbuf, dtype, shape = raw
    return np.frombuffer(rawbuf, dtype=dtype).reshape(shape)
