"""
redrock.utils
=============

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
        ### return data.byteswap().newbyteorder()  # only works with numpy<2
        return data.byteswap().view(data.dtype.newbyteorder('native'))  # works with numpy 1.x or 2.x


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
    import tty, termios
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

    If the input array has size=0, the original is returned without
    wrapping it in a size=0 multiprocessing.RawArray (which generates
    warnings)
    """
    import multiprocessing as mp

    # A zero-length array generates the warning documented in
    # https://github.com/desihub/redrock/issues/250; capture it.
    if original.size == 0:
        return original
    
    typecode = original.dtype.char
    shape = original.shape

    raw = mp.RawArray(typecode, original.ravel())
    nd = np.array(raw, dtype=typecode, copy=False).view()
    nd.shape = shape
    return nd

def distribute_work(nproc, ids, weights=None, capacities=None):
    """Helper function to distribute work among processes with varying capacities.

    Args:
        nproc (int): the number of processes.
        ids (list): list of work unit IDs
        weights (dict): dictionary of weights for each ID. If None,
            use equal weighting.
        capacities (list): list of process capacities. If None,
            use equal capacity per process. A process with higher capacity
            can handle more work.

    Returns:
        list: A list (one element for each process) with each element
            being a list of the IDs assigned to that process.

    """
    # Sort ids by weights (descending)
    if weights is None:
        weights = { x : 1 for x in ids }
    sids = list(sorted(ids, key=lambda x: weights[x], reverse=True))

    # If capacities are not provided, assume they are equal
    if capacities is None:
        capacities = [1] * nproc

    # Initialize distributed list of ids
    dist = [list() for _ in range(nproc)]

    # Initialize process list. Processes are modeled using dictionary
    # with fields for a unique id, capacity, and load (total weight of work).
    processes = [dict(id=i, capacity=c, load=0) for i, c in enumerate(capacities)]

    for id in sids:
        w = weights[id]
        # Identify process to receive task. Smallest normalized load, break ties with capacity, followed by id.
        minload = min(processes, key=lambda p: ((p['load'] + w)/p['capacity'], 1/p['capacity'], p['id']))
        i = processes.index(minload)
        # Assign work unit to process
        minload['load'] += weights[id]
        dist[i].append(id)

    return dist

def getGPUCountMPI(comm):
    """Determine the number of GPUs available when running in MPI mode

    In MPI mode, cupy.cuda.runtime.getDeviceCount() always returns 1 so
    this method determines the actual number of GPUs.

    Args:
        comm: the MPI communicator

    Returns:
        int: the number of GPUs
    """
    #First check /proc because this is fast
    if os.access('/proc/driver/nvidia/gpus/', os.F_OK):
        return len(os.listdir('/proc/driver/nvidia/gpus/'))
    #Now gather device IDs which is slow
    if comm.rank == 0:
        print("WARNING:  --max-gpuprocs option was not specified")
        print("WARNING:  Cannot find /proc/driver/nvidia/gpus/")
        print("WARNING:  Finding number of devices by gathering PCI Bus ids")
    import cupy
    #Get this device's PCI id
    n = cupy.cuda.runtime.getDevice()
    pci_id = cupy.cuda.runtime.deviceGetPCIBusId(n)
    #Gather and remove duplicates
    pci_id_list = comm.allgather(pci_id)
    pci_id_list = list(set(pci_id_list))
    if comm.rank == 0:
        print("WARNING:  Found {:d} GPUs".format(len(pci_id_list)))
    return len(pci_id_list)

def reduced_wavelength(wave):
    """function to calculate reduced wavelength arrya for 
    legendre polynomial in archetype mode, 
    legendre polynomials are orthogonal in [-1,1]
    Args:
        wave (array): wavelength for which reduced wavelengths to be estimated
    Return:
        reduced wavelength in [-1,1] range
    """
    wavemax = wave.max()
    wavemin = wave.min()
    return 2*(wave - wavemin) / (wavemax - wavemin) - 1
