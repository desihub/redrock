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

class Inoue14(object):
    def __init__(self, scale_tau=1.):
        """
        IGM absorption from Inoue et al. (2014)
        
        Parameters
        ----------
        scale_tau : float
            Parameter multiplied to the IGM :math:`\tau` values (exponential 
            in the linear absorption fraction).  
            I.e., :math:`f_\mathrm{igm} = e^{-\mathrm{scale\_tau} \tau}`.

        Copyright (c) 2016-2022 Gabriel Brammer

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        
        """
        super(Inoue14, self).__init__()
        
        self._load_data()
        self.scale_tau = scale_tau
        self.has_gpu_data = False

    @staticmethod
    def _pow(a, b):
        """C-like power, a**b
        """
        return a**b
    
    def _load_data(self):
        from importlib import resources
        LAF_file = str(resources.files('redrock').joinpath('data/LAFcoeff.txt'))
        DLA_file = str(resources.files('redrock').joinpath('data/DLAcoeff.txt'))
    
        data = np.loadtxt(LAF_file, unpack=True)
        ix, lam, ALAF1, ALAF2, ALAF3 = data
        self.lam = lam[:,np.newaxis]
        self.ALAF1 = ALAF1[:,np.newaxis]
        self.ALAF2 = ALAF2[:,np.newaxis]
        self.ALAF3 = ALAF3[:,np.newaxis]
        
        data = np.loadtxt(DLA_file, unpack=True)
        ix, lam, ADLA1, ADLA2 = data
        self.ADLA1 = ADLA1[:,np.newaxis]
        self.ADLA2 = ADLA2[:,np.newaxis]
                
        return True

    def copy_data_to_gpu(self):
        if (self.has_gpu_data):
            return
        import cupy as cp
        self.gpu_ALAF1 = cp.asarray(self.ALAF1[:,:,None])
        self.gpu_ALAF2 = cp.asarray(self.ALAF2[:,:,None])
        self.gpu_ALAF3 = cp.asarray(self.ALAF3[:,:,None])
        self.gpu_ADLA1 = cp.asarray(self.ADLA1[:,:,None])
        self.gpu_ADLA2 = cp.asarray(self.ADLA2[:,:,None])
        self.gpu_lam = cp.asarray(self.lam[:,:,None])
        self.has_gpu_data = True
        return

    @property
    def NA(self):
        """
        Number of Lyman-series lines
        """
        return self.lam.shape[0]

    def tLSLAF_old(self, zS, lobs):
        """
        Lyman series, Lyman-alpha forest
        """
        z1LAF = 1.2
        z2LAF = 4.7

        l2 = self.lam #[:, np.newaxis]
        tLSLAF_value = np.zeros_like(lobs*l2).T

        x0 = (lobs < l2*(1+zS))
        x1 = x0 & (lobs < l2*(1+z1LAF))
        x2 = x0 & ((lobs >= l2*(1+z1LAF)) & (lobs < l2*(1+z2LAF)))
        x3 = x0 & (lobs >= l2*(1+z2LAF))

        tLSLAF_value = np.zeros_like(lobs*l2)
        tLSLAF_value[x1] += ((self.ALAF1/l2**1.2)*lobs**1.2)[x1]
        tLSLAF_value[x2] += ((self.ALAF2/l2**3.7)*lobs**3.7)[x2]
        tLSLAF_value[x3] += ((self.ALAF3/l2**5.5)*lobs**5.5)[x3]

        return tLSLAF_value.sum(axis=0)


    def tLSDLA_old(self, zS, lobs):
        """
        Lyman Series, DLA
        """
        z1DLA = 2.0

        l2 = self.lam #[:, np.newaxis]
        tLSDLA_value = np.zeros_like(lobs*l2)

        x0 = (lobs < l2*(1+zS)) & (lobs < l2*(1.+z1DLA))
        x1 = (lobs < l2*(1+zS)) & ~(lobs < l2*(1.+z1DLA))

        tLSDLA_value[x0] += ((self.ADLA1/l2**2)*lobs**2)[x0]
        tLSDLA_value[x1] += ((self.ADLA2/l2**3)*lobs**3)[x1]

        return tLSDLA_value.sum(axis=0)

    def tLCDLA_old(self, zS, lobs):
        """
        Lyman continuum, DLA
        """
        z1DLA = 2.0
        lamL = 911.8

        tLCDLA_value = np.zeros_like(lobs)

        x0 = lobs < lamL*(1.+zS)
        if zS < z1DLA:
            tLCDLA_value[x0] = 0.2113 * self._pow(1.0+zS, 2) - 0.07661 * self._pow(1.0+zS, 2.3) * self._pow(lobs[x0]/lamL, (-3e-1)) - 0.1347 * self._pow(lobs[x0]/lamL, 2)
        else:
            x1 = lobs >= lamL*(1.+z1DLA)

            tLCDLA_value[x0 & x1] = 0.04696 * self._pow(1.0+zS, 3) - 0.01779 * self._pow(1.0+zS, 3.3) * self._pow(lobs[x0 & x1]/lamL, (-3e-1)) - 0.02916 * self._pow(lobs[x0 & x1]/lamL, 3)
            tLCDLA_value[x0 & ~x1] =0.6340 + 0.04696 * self._pow(1.0+zS, 3) - 0.01779 * self._pow(1.0+zS, 3.3) * self._pow(lobs[x0 & ~x1]/lamL, (-3e-1)) - 0.1347 * self._pow(lobs[x0 & ~x1]/lamL, 2) - 0.2905 * self._pow(lobs[x0 & ~x1]/lamL, (-3e-1))

        return tLCDLA_value


    def tLCLAF_old(self, zS, lobs):
        """
        Lyman continuum, LAF
        """
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = np.zeros_like(lobs)

        x0 = lobs < lamL*(1.+zS)

        if zS < z1LAF:
            tLCLAF_value[x0] = 0.3248 * (self._pow(lobs[x0]/lamL, 1.2) - self._pow(1.0+zS, -9e-1) * self._pow(lobs[x0]/lamL, 2.1))
        elif zS < z2LAF:
            x1 = lobs >= lamL*(1+z1LAF)
            tLCLAF_value[x0 & x1] = 2.545e-2 * (self._pow(1.0+zS, 1.6) * self._pow(lobs[x0 & x1]/lamL, 2.1) - self._pow(lobs[x0 & x1]/lamL, 3.7))
            tLCLAF_value[x0 & ~x1] = 2.545e-2 * self._pow(1.0+zS, 1.6) * self._pow(lobs[x0 & ~x1]/lamL, 2.1) + 0.3248 * self._pow(lobs[x0 & ~x1]/lamL, 1.2) - 0.2496 * self._pow(lobs[x0 & ~x1]/lamL, 2.1)
        else:
            x1 = lobs > lamL*(1.+z2LAF)
            x2 = (lobs >= lamL*(1.+z1LAF)) & (lobs < lamL*(1.+z2LAF))
            x3 = lobs < lamL*(1.+z1LAF)

            tLCLAF_value[x0 & x1] = 5.221e-4 * (self._pow(1.0+zS, 3.4) * self._pow(lobs[x0 & x1]/lamL, 2.1) - self._pow(lobs[x0 & x1]/lamL, 5.5))
            tLCLAF_value[x0 & x2] = 5.221e-4 * self._pow(1.0+zS, 3.4) * self._pow(lobs[x0 & x2]/lamL, 2.1) + 0.2182 * self._pow(lobs[x0 & x2]/lamL, 2.1) - 2.545e-2 * self._pow(lobs[x0 & x2]/lamL, 3.7)
            tLCLAF_value[x0 & x3] = 5.221e-4 * self._pow(1.0+zS, 3.4) * self._pow(lobs[x0 & x3]/lamL, 2.1) + 0.3248 * self._pow(lobs[x0 & x3]/lamL, 1.2) - 3.140e-2 * self._pow(lobs[x0 & x3]/lamL, 2.1)

        return tLCLAF_value


    def tLSLAF(self, zS, lobs, use_gpu=False):
        """
        Lyman series, Lyman-alpha forest
        """
        if (use_gpu):
            import cupy as cp
            ALAF1 = self.gpu_ALAF1
            ALAF2 = self.gpu_ALAF2
            ALAF3 = self.gpu_ALAF3
            l2 = self.gpu_lam
            zS = cp.asarray(zS)
        else:
            ALAF1 = self.ALAF1[:,:,None]
            ALAF2 = self.ALAF2[:,:,None]
            ALAF3 = self.ALAF3[:,:,None]
            l2 = self.lam[:,:,None]

        z1LAF = 1.2
        z2LAF = 4.7

        tLSLAF_value = np.zeros_like(lobs*l2).T

        x0 = (lobs / (1+zS)[:,None]) < l2
        x1 = x0 & (lobs < l2*(1+z1LAF))
        x2 = x0 & ((lobs >= l2*(1+z1LAF)) & (lobs < l2*(1+z2LAF)))
        x3 = x0 & (lobs >= l2*(1+z2LAF))

        tLSLAF_value = np.zeros_like(lobs*l2)
        tLSLAF_value[x1] += ((ALAF1/l2**1.2)*lobs**1.2)[x1]
        tLSLAF_value[x2] += ((ALAF2/l2**3.7)*lobs**3.7)[x2]
        tLSLAF_value[x3] += ((ALAF3/l2**5.5)*lobs**5.5)[x3]

        return tLSLAF_value.sum(axis=0)


    def tLSDLA(self, zS, lobs, use_gpu=False):
        """
        Lyman Series, DLA
        """
        if (use_gpu):
            import cupy as cp
            ADLA1 = self.gpu_ADLA1
            ADLA2 = self.gpu_ADLA2
            l2 = self.gpu_lam
            zS = cp.asarray(zS)
        else:
            ADLA1 = self.ADLA1[:,:,None]
            ADLA2 = self.ADLA2[:,:,None]
            l2 = self.lam[:,:,None]
        z1DLA = 2.0
        
        tLSDLA_value = np.zeros_like(lobs*l2)
        
        x0 = ((lobs / (1+zS)[:,None]) < l2) & (lobs < l2*(1.+z1DLA))
        x1 = ((lobs / (1+zS)[:,None]) < l2) & ~(lobs < l2*(1.+z1DLA))
        
        tLSDLA_value[x0] += ((ADLA1/l2**2)*lobs**2)[x0]
        tLSDLA_value[x1] += ((ADLA2/l2**3)*lobs**3)[x1]
                
        return tLSDLA_value.sum(axis=0)


    def tLCDLA(self, zS, lobs, use_gpu=False):
        """
        Lyman continuum, DLA
        """
        if (use_gpu):
            import cupy as cp
            power = cp.power
            zS = cp.asarray(zS)
        else:
            power = np.power
        z1DLA = 2.0
        lamL = 911.8
        
        tLCDLA_value = np.zeros_like(lobs)
        
        x0 = lobs < lamL*(1.+zS)[:,None]

        y0 = x0 & (zS[:,None] < z1DLA)
        tLCDLA_value[y0] = 0.2113 * (power(1.0+zS[:,None], 2)*y0)[y0] - 0.07661 * (power(1.0+zS[:,None], 2.3)*y0)[y0] * power(lobs[y0]/lamL, (-3e-1)) - 0.1347 * power(lobs[y0]/lamL, 2)

        y0 = x0 & (zS[:,None] >= z1DLA)
        x1 = lobs >= lamL*(1.+z1DLA)
        tLCDLA_value[y0 & x1] = 0.04696 * (power(1.0+zS[:,None], 3)*y0)[y0 & x1] - 0.01779 * (power(1.0+zS[:,None], 3.3)*y0)[y0 & x1] * power(lobs[y0 & x1]/lamL, (-3e-1)) - 0.02916 * power(lobs[y0 & x1]/lamL, 3)
        tLCDLA_value[y0 & ~x1] =0.6340 + 0.04696 * (power(1.0+zS[:,None], 3)*y0)[y0 & ~x1] - 0.01779 * (power(1.0+zS[:,None], 3.3)*y0)[y0 & ~x1] * power(lobs[y0 & ~x1]/lamL, (-3e-1)) - 0.1347 * power(lobs[y0 & ~x1]/lamL, 2) - 0.2905 * power(lobs[y0 & ~x1]/lamL, (-3e-1))
        
        return tLCDLA_value


    def tLCLAF(self, zS, lobs, use_gpu=False):
        """
        Lyman continuum, LAF
        """
        if (use_gpu):
            import cupy as cp
            power = cp.power
            zS = cp.asarray(zS)
        else:
            power = np.power
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = np.zeros_like(lobs)
        
        x0 = lobs < lamL*(1.+zS)[:,None]
        y0 = x0 & (zS[:,None] < z1LAF) 
        tLCLAF_value[y0] = 0.3248 * (power(lobs[y0]/lamL, 1.2) - (power(1.0+zS[:,None], -9e-1)*y0)[y0] * power(lobs[y0]/lamL, 2.1))

        y0 = x0 & (zS[:,None] < z2LAF)
        x1 = lobs >= lamL*(1+z1LAF)
        tLCLAF_value[y0 & x1] = 2.545e-2 * ((power(1.0+zS[:,None], 1.6)*y0)[y0 & x1] * power(lobs[y0 & x1]/lamL, 2.1) - power(lobs[y0 & x1]/lamL, 3.7))
        tLCLAF_value[y0 & ~x1] = 2.545e-2 * (power(1.0+zS[:,None], 1.6)*y0)[y0 & ~x1] * power(lobs[y0 & ~x1]/lamL, 2.1) + 0.3248 * power(lobs[y0 & ~x1]/lamL, 1.2) - 0.2496 * power(lobs[y0 & ~x1]/lamL, 2.1)

        y0 = x0 & (zS[:,None] >= z2LAF)
        x1 = lobs > lamL*(1.+z2LAF)
        x2 = (lobs >= lamL*(1.+z1LAF)) & (lobs < lamL*(1.+z2LAF))
        x3 = lobs < lamL*(1.+z1LAF)

        tLCLAF_value[y0 & x1] = 5.221e-4 * ((power(1.0+zS[:,None], 3.4)*y0)[y0 & x1] * power(lobs[y0 & x1]/lamL, 2.1) - power(lobs[y0 & x1]/lamL, 5.5))
        tLCLAF_value[y0 & x2] = 5.221e-4 * (power(1.0+zS[:,None], 3.4)*y0)[y0 & x2] * power(lobs[y0 & x2]/lamL, 2.1) + 0.2182 * power(lobs[y0 & x2]/lamL, 2.1) - 2.545e-2 * power(lobs[y0 & x2]/lamL, 3.7)
        tLCLAF_value[y0 & x3] = 5.221e-4 * (power(1.0+zS[:,None], 3.4)*y0)[y0 & x3] * power(lobs[y0 & x3]/lamL, 2.1) + 0.3248 * power(lobs[y0 & x3]/lamL, 1.2) - 3.140e-2 * power(lobs[y0 & x3]/lamL, 2.1)
            
        return tLCLAF_value


    def full_IGM_old(self, z, lobs):
        """Get full Inoue IGM absorption
        
        Parameters
        ----------
        z : float
            Redshift to evaluate IGM absorption
        
        lobs : array
            Observed-frame wavelength(s) in Angstroms.
        
        Returns
        -------
        abs : array
            IGM absorption
        
        """
        tau_LS = self.tLSLAF_old(z, lobs) + self.tLSDLA_old(z, lobs)
        tau_LC = self.tLCLAF_old(z, lobs) + self.tLCDLA_old(z, lobs)

        ### Upturn at short wavelengths, low-z
        #k = 1./100
        #l0 = 600-6/k
        #clip = lobs/(1+z) < 600.
        #tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip = 0.
        
        return np.exp(-self.scale_tau*(tau_LC + tau_LS + tau_clip))

    def full_IGM(self, z, lobs, use_gpu=False):
        """Get full Inoue IGM absorption

        Parameters
        ----------
        z : float array
            Redshift to evaluate IGM absorption

        lobs : array
            Observed-frame wavelength(s) in Angstroms.

        Returns
        -------
        abs : array
            IGM absorption

        """

        if (use_gpu):
            import cupy as cp
            arrexp = cp.exp
            self.copy_data_to_gpu()
        else:
            arrexp = np.exp

        tau_LS = self.tLSLAF(z, lobs, use_gpu=use_gpu) + self.tLSDLA(z, lobs, use_gpu=use_gpu)
        tau_LC = self.tLCLAF(z, lobs, use_gpu=use_gpu) + self.tLCDLA(z, lobs, use_gpu=use_gpu)

        ### Upturn at short wavelengths, low-z
        #k = 1./100
        #l0 = 600-6/k
        #clip = lobs/(1+z) < 600.
        #tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip = 0.

        return arrexp(-self.scale_tau*(tau_LC + tau_LS + tau_clip))

    def build_grid(self, zgrid, lrest):
        """Build a spline interpolation object for fast IGM models
        
        Returns: self.interpolate
        """
        
        from scipy.interpolate import CubicSpline
        igm_grid = np.zeros((len(zgrid), len(lrest)))
        for iz in range(len(zgrid)):
            igm_grid[iz,:] = self.full_IGM(zgrid[iz], lrest*(1+zgrid[iz]))
        
        self.interpolate = CubicSpline(zgrid, igm_grid)

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

    """
    import multiprocessing as mp

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

IGM = None

def transmission_IGM_old(zObj, lObs, use_gpu=False):
    """Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)

    This method will handle 3 options:
    1 -> For GPU mode, zObj is an array of all z for a template and the return
    value will be a cupy array (nz x nlambda)
    2-> In CPU mode, it can auto-detect if zObj is a numpy array and if so,
    again, all z will be processed as a vector and the return value
    will be a numpy array (nz x nlambda)
    3-> For legacy, it is still supported to pass zObj as a float and
    in this case, the return value will be a 1-d numpy array (nlambda).

    Args:
        zObj (float or array of float): Redshift(s) of object
        lObs (array of float): wavelength grid
        use_gpu (boolean): whether to use CUPY
    Returns:
        array of float: transmitted flux fraction (nlambda in case of
        scalar input; nz x nlambda in case of array input)

    """
    global IGM

    if IGM is None:
        IGM = Inoue14()

    if use_gpu:
        import cupy as cp
        tile = cp.tile
        asarray = cp.asarray
    else:
        tile = np.tile
        asarray = np.asarray

    #Lyman_series = constants.Lyman_series
    min_wave = 0
    if (np.isscalar(zObj)):
        #zObj is a float
        lRF = lObs/(1.+zObj)
    else:
        if (len(zObj) == 0):
            #Empty z array
            return np.ones((0, len(lObs)), dtype=np.float64)
        #This is an array of float
        min_wave = lObs.min()/(1+zObj.max())
        #if (lObs.min()/(1+zObj.max()) > Lyman_series['Lya']['line']):
        #if (min_wave > Lyman_series['Lya']['line']):
        #    #Return None if wavelength range doesn't overlap with Lyman series
        #    #No need to perform any calculations in this case
        #    return None
        if (not use_gpu and type(zObj) != np.ndarray):
            #Cupy array passed??
            zObj = zObj.get()
        lObs = tile(lObs, (zObj.size, 1))
        lRF = lObs/(1.+asarray(zObj)[:,None])
    if use_gpu:
        T = np.ones_like(lRF.get()) # .get() hack
    else:
        T = np.ones_like(lRF)
    for iz, myz in enumerate(zObj):
        if myz < 2.:
            continue
        if use_gpu:
            T[iz, :] = IGM.full_IGM_old(myz, lRF[iz, :].get()) # .get() hack
        else:
            T[iz, :] = IGM.full_IGM_old(myz, lRF[iz, :]) # .get() hack
    if use_gpu:
        T = asarray(T) # hack
    ##Lyman_series = constants.Lyman_series
    #min_wave = 0
    #if np.isscalar(zObj):
    #    #zObj is a float
    #    lRF = lObs/(1.+zObj)
    #else:
    #    if (len(zObj) == 0):
    #        #Empty z array
    #        return np.ones((0, len(lObs)), dtype=np.float64)
    #    #This is an array of float
    #    min_wave = lObs.min()/(1+zObj.max())
    #    #if (lObs.min()/(1+zObj.max()) > Lyman_series['Lya']['line']):
    #    #if (min_wave > Lyman_series['Lya']['line']):
    #    #    #Return None if wavelength range doesn't overlap with Lyman series
    #    #    #No need to perform any calculations in this case
    #    #    return None
    #    if (not use_gpu and type(zObj) != np.ndarray):
    #        #Cupy array passed??
    #        zObj = zObj.get()
    #    lObs = tile(lObs, (zObj.size, 1))
    #    lRF = lObs/(1.+asarray(zObj)[:,None])

    T = tile(np.ones_like(lObs), (zObj.size, 1))

    W = np.where(lObs.min() / (1. + zObj) < 1215.67)[0]
    if len(W) > 0:
        for iz in W:
            T[iz, :] = asarray(IGM.full_IGM(zObj[iz], lObs))

    #for l in list(Lyman_series.keys()):
    #    if (min_wave > Lyman_series[l]['line']):
    #        continue
    #    w      = lRF<Lyman_series[l]['line']
    #    zpix   = lObs[w]/Lyman_series[l]['line']-1.
    #    tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
    #    T[w]  *= np.exp(-tauEff)
    
    #if (np.isscalar(zObj) and use_gpu):
    #    T = asarray(T)
        
    return T

def transmission_IGM(zObj, lObs, use_gpu=False):
    """Calculate the transmitted flux fraction from the Lyman series
    and due to the IGM.  This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)

    This method will handle 3 options:
    1 -> For GPU mode, zObj is an array of all z for a template and the return
    value will be a cupy array (nz x nlambda)
    2-> In CPU mode, it can auto-detect if zObj is a numpy array and if so,
    again, all z will be processed as a vector and the return value
    will be a numpy array (nz x nlambda)
    3-> For legacy, it is still supported to pass zObj as a float and
    in this case, the return value will be a 1-d numpy array (nlambda).

    Args:
        zObj (float or array of float): Redshift(s) of object
        lObs (array of float): wavelength grid
        use_gpu (boolean): whether to use CUPY
    Returns:
        array of float: transmitted flux fraction (nlambda in case of
        scalar input; nz x nlambda in case of array input)

    """
    global IGM
    if IGM is None:
        IGM = Inoue14()

    if use_gpu:
        import cupy as cp
        tile = cp.tile
        asarray = cp.asarray
    else:
        tile = np.tile
        asarray = np.asarray

    min_wave = 0
    if np.isscalar(zObj):
        #zObj is a float
        min_wave = np.array([lObs.min()/(1.+zObj)])
    else:
        if len(zObj) == 0:
            #Empty z array
            return np.ones((0, len(lObs)), dtype=np.float64)
        #This is an array of float
        min_wave = lObs.min()/(1+zObj.max())
        if min_wave > 1220.: 
            #Return None if wavelength range doesn't overlap with Lyman series
            #No need to perform any calculations in this case
            return None
        #Calculate min wave for every z
        min_wave = lObs.min()/(1+zObj)
        if not use_gpu and type(zObj) != np.ndarray:
            #Cupy array passed??
            zObj = zObj.get()
        lObs = tile(lObs, (zObj.size, 1))
    T = np.ones_like(lObs)
    # Only process wavelengths at or shorter than Lya. Now, Lya is at 1215.67,
    # but let the IGM class figure out the exact transmission around the line.
    i = min_wave < 1220.
    T[i, :] = IGM.full_IGM(asarray(zObj[i]), lObs[i,:], use_gpu=use_gpu)
    if np.isscalar(zObj) and use_gpu:
        T = asarray(T)
    return T


def transmission_Lyman(zObj,lObs, use_gpu=False):
    """Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)

    This method will handle 3 options:
    1 -> For GPU mode, zObj is an array of all z for a template and the return
    value will be a cupy array (nz x nlambda)
    2-> In CPU mode, it can auto-detect if zObj is a numpy array and if so,
    again, all z will be processed as a vector and the return value
    will be a numpy array (nz x nlambda)
    3-> For legacy, it is still supported to pass zObj as a float and
    in this case, the return value will be a 1-d numpy array (nlambda).

    Args:
        zObj (float or array of float): Redshift(s) of object
        lObs (array of float): wavelength grid
        use_gpu (boolean): whether to use CUPY
    Returns:
        array of float: transmitted flux fraction (nlambda in case of
        scalar input; nz x nlambda in case of array input)

    """
    # Inoue14 models
    #return transmission_IGM(zObj,lObs, use_gpu)
    #return transmission_IGM_old(zObj,lObs, use_gpu)

    # original Calura/Irsic/Kamble form
    if (use_gpu):
        import cupy as cp
        tile = cp.tile
        asarray = cp.asarray
    else:
        tile = np.tile
        asarray = np.asarray

    Lyman_series = constants.Lyman_series
    min_wave = 0
    if (np.isscalar(zObj)):
        #zObj is a float
        lRF = lObs/(1.+zObj)
    else:
        if (len(zObj) == 0):
            #Empty z array
            return np.ones((0, len(lObs)), dtype=np.float64)
        #This is an array of float
        min_wave = lObs.min()/(1+zObj.max())
        #if (lObs.min()/(1+zObj.max()) > Lyman_series['Lya']['line']):
        if (min_wave > Lyman_series['Lya']['line']):
            #Return None if wavelength range doesn't overlap with Lyman series
            #No need to perform any calculations in this case
            return None
        if (not use_gpu and type(zObj) != np.ndarray):
            #Cupy array passed??
            zObj = zObj.get()
        lObs = tile(lObs, (zObj.size, 1))
        lRF = lObs/(1.+asarray(zObj)[:,None])
    T = np.ones_like(lRF)
    for l in list(Lyman_series.keys()):
        if (min_wave > Lyman_series[l]['line']):
            continue
        w      = lRF<Lyman_series[l]['line']
        zpix   = lObs[w]/Lyman_series[l]['line']-1.
        tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
        T[w]  *= np.exp(-tauEff)
    if (np.isscalar(zObj) and use_gpu):
        T = asarray(T)
    return T

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
