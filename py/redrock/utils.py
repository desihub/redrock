"""
Redrock utility functions.
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import time

import numpy as np

from . import constants

#- From https://github.com/desihub/desispec io.util.native_endian
def native_endian(data):
    """Convert numpy array data to native endianness if needed.

    Returns new array if endianness is swapped, otherwise returns input data

    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.

    Args:
        data (array): input array

    Returns:
        array: original array if input in native endianness, otherwise a copy
            with the bytes swapped.

    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()


def encode_column(c):
    """Returns a bytes column encoded into a string column.

    Args:
        c (Table column): a column of a Table.

    Returns:
        array: an array of strings.

    """
    return c.astype((str, c.dtype.itemsize))


#- Adapted from http://stackoverflow.com/a/21659588; unix only
def getch():
    """Return a single character from stdin.
    """
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def elapsed(timer, prefix, comm=None):
    """Get and print the elapsed time.

    If timer is None, compute the start time and return.  Otherwise, find the
    elapsed time and print a message before returning the new start time.

    Args:
        timer (float): time in seconds for some arbitrary epoch.  If "None",
            get the current time and return.
        prefix (str): string to print before the elapsed time.
        comm (mpi4py.MPI.Comm): optional communicator.

    Returns:
        float: the new start time in seconds.

    """
    if comm is not None:
        comm.barrier()

    cur = time.time()
    if timer is not None:
        elapsed = cur - timer
        if ((comm is None) or (comm.rank == 0)):
            print("{}: {:0.1f} seconds".format(prefix, elapsed))
            sys.stdout.flush()

    return cur


def nersc_login_node():
    """Returns True if we are on a NERSC login node, else False.
    """
    if ("NERSC_HOST" in os.environ) and ("SLURM_JOB_NAME" not in os.environ):
        return True
    else:
        return False


def get_mp(requested):
    """Returns a reasonable number of multiprocessing processes.

    This checks whether the requested value makes sense, and also whether we
    are running on a NERSC login node (and hence would get in trouble trying
    to use all cores).

    Args:
        requested (int): the requested number of processes.

    Returns:
        int: the number of processes to use.

    """
    import multiprocessing as mp

    mpmax = mp.cpu_count()
    mpbest = mpmax // 2
    if mpbest == 0:
        mpbest = 1

    procs = None
    if requested == 0:
        procs = mpbest
    elif requested > mpmax:
        print("Requested number of processes ({}) is too large, "
            "reducing to {}".format(requested, mpmax))
        sys.stdout.flush()
        procs = mpmax
    else:
        procs = requested

    # On NERSC login nodes, avoid hogging the whole node (and getting an
    # unhappy phone call).
    login_max = 4
    if nersc_login_node():
        if procs > login_max:
            print("Running on a NERSC login node- reducing number of processes"
                " to {}".format(login_max))
            sys.stdout.flush()
            procs = login_max

    return procs


def mp_array(original):
    """Allocate a raw shared memory buffer and wrap it in an ndarray.

    This allocates a multiprocessing.RawArray and wraps the buffer
    with an ndarray.

    Args:
        typcode (str): the type code of the array.
        size_or_init: passed to the RawArray constructor.

    Returns;
        ndarray: the wrapped data.

    """
    import multiprocessing as mp

    typecode = original.dtype.char
    shape = original.shape

    raw = mp.RawArray(typecode, original.ravel())
    nd = np.array(raw, dtype=typecode, copy=False).view()
    nd.shape = shape
    return nd


def distribute_work(nproc, ids, weights=None):
    """Helper function to distribute work among processes.

    This takes a list of unique IDs associated with each work unit, and a
    dictionary of weights for each ID.  It returns a list of lists which
    contain the IDs assigned to each process.

    Args:
        nproc (int): the number of processes.
        ids (list): list of IDs
        weights (dict): dictionary of weights for each ID.  If None,
            use equal weighting.

    Returns:
        list:  A list (one element for each process) with each element
            being a list of the IDs assigned to that process.

    """
    #
    # These are two helper functions.
    #
    def distribute_required_groups(A, max_per_group):
        ngroup = 1
        total = 0
        for i in range(A.shape[0]):
            total += A[i]
            if total > max_per_group:
                total = A[i]
                ngroup += 1
        return ngroup

    def distribute_partition(A, k):
        low = np.max(A)
        high = np.sum(A)
        while low < high:
            mid = low + int((high - low) / 2)
            required = distribute_required_groups(A, mid)
            if required <= k:
                high = mid
            else:
                low = mid + 1
        return low

    # First sort the IDs and weights

    if weights is None:
        weights = { x : 1 for x in ids }

    sids = list(sorted(ids))
    wts = np.array([ weights[x] for x in sids ], dtype=np.float64)

    # Compute the partitioning

    max_per_proc = float(distribute_partition(wts.astype(np.int64),
        nproc))

    if len(sids) <= nproc:
        # This is wasteful, but the best we can do is assign one target
        # to each process and leave some idle.
        dist = [ [x] for x in sids ]
        dist.extend([ list() for i in range(nproc - len(sids)) ])
        return dist

    goal = np.sum(wts) / float(nproc)

    ranges = []

    off = 0
    curweight = 0.0
    for cur in range(0, len(wts)):
        if curweight + wts[cur] > max_per_proc:
            ranges.append( (off, cur-off) )
            over = curweight - goal
            curweight = wts[cur] + over
            off = cur
        else:
            curweight += wts[cur]

    # Now distribute the remaining items uniformly among the remaining
    # processes.  In the case of good load balance, there should only be
    # one worker left, but that does not always happen...

    remain_procs = nproc-len(ranges)
    remain_items = len(wts) - off

    if remain_items <= remain_procs:
        # really bad load imbalance...
        ranges.extend([ (off+i, 1) for i in range(remain_items) ])
    else:
        for i in range(remain_procs):
            ntask = remain_items // remain_procs
            firsttask = 0
            leftover = remain_items % remain_procs
            if i < leftover:
                ntask += 1
                firsttask = i * ntask
            else:
                firsttask = ((ntask + 1) * leftover) + \
                    (ntask * (i - leftover))
            ranges.append( (off+firsttask, ntask) )

    # Convert our ranges into a list of IDs for each process

    dist = [ list(sids[x[0]:x[0]+x[1]]) for x in ranges ]
    if len(dist) < nproc:
        empty = nproc - len(dist)
        for i in range(empty):
            dist.append( list() )

    return dist


def transmission_Lyman(zObj,lObs):
    """Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
        1 -> everything is transmitted (medium is transparent)
        0 -> nothing is transmitted (medium is opaque)
    Args:
        zObj (float): Redshift of object
        lObs (array of float): wavelength grid
    Returns:
        array of float: transmitted flux fraction
    """

    lRF = lObs/(1.+zObj)
    T   = np.ones(lObs.size)

    Lyman_series = constants.Lyman_series
    for l in list(Lyman_series.keys()):
        w      = lRF<Lyman_series[l]['line']
        zpix   = lObs[w]/Lyman_series[l]['line']-1.
        tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
        T[w]  *= np.exp(-tauEff)

    return T
def ivar_Lyman(zObj,lObs,ivarPipe,capValue=50.):
    """Cap the inverse variance in the Lyman series
    Args:
        zObj     (float): Redshift of object
        lObs     (array of float): wavelength grid
        ivarPipe (array of float): pipeline inverse variance
    Returns:
        array of float: capped inverse variance
    """

    lRF  = lObs/(1.+zObj)
    ivar = ivarPipe.copy()

    Lyman_series = constants.Lyman_series
    for l in list(Lyman_series.keys()):
        w  = lRF<Lyman_series[l]['line']
        w &= ivar>capValue
        ivar[w] = capValue

    return ivar
