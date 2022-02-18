"""
Classes and functions for templates.
"""

from __future__ import absolute_import, division, print_function

import sys
from glob import glob
import os
import traceback

import numpy as np
from astropy.io import fits

from .utils import native_endian, elapsed, transmission_Lyman

from .rebin import rebin_template, trapz_rebin


class Template(object):
    """A spectral Template PCA object.

    The template data is read from a redrock-format template file.
    Alternatively, the data can be specified in the constructor.

    Args:
        filename (str): the path to the template file, either absolute or
            relative to the RR_TEMPLATE_DIR environment variable.

    """
    def __init__(self, filename=None, spectype=None, redshifts=None,
        wave=None, flux=None, subtype=None):

        if filename is not None:
            fx = None
            if os.path.exists(filename):
                fx = fits.open(filename, memmap=False)
            else:
                xfilename = os.path.join(os.getenv('RR_TEMPLATE_DIR'), filename)
                if os.path.exists(xfilename):
                    fx = fits.open(xfilename, memmap=False)
                else:
                    raise IOError('unable to find '+filename)

            hdr = fx['BASIS_VECTORS'].header
            if 'VERSION' in hdr:
                self._version = hdr['VERSION']
            else:
                self._version = 'unknown'

            self.wave = np.asarray(hdr['CRVAL1'] + \
                hdr['CDELT1']*np.arange(hdr['NAXIS1']), dtype=np.float64)
            if 'LOGLAM' in hdr and hdr['LOGLAM'] != 0:
                self.wave = 10**self.wave

            self.flux = np.asarray(native_endian(fx['BASIS_VECTORS'].data),
                dtype=np.float64)

            self._redshifts = None

            ## find out if redshift info is present in the file
            old_style_templates = True
            try:
                self._redshifts = native_endian(fx['REDSHIFTS'].data)
                old_style_templates = False
            except KeyError:
                pass

            fx.close()

            self._rrtype = hdr['RRTYPE'].strip().upper()
            if old_style_templates:
                if self._rrtype == 'GALAXY':
                    # redshifts = 10**np.arange(np.log10(1+0.005),
                    # np.log10(1+2.0), 1.5e-4) - 1
                    self._redshifts = 10**np.arange(np.log10(1-0.005),
                        np.log10(1+1.7), 3e-4) - 1
                elif self._rrtype == 'STAR':
                    self._redshifts = np.arange(-0.002, 0.00201, 4e-5)
                elif self._rrtype == 'QSO':
                    self._redshifts = 10**np.arange(np.log10(1+0.05),
                        np.log10(1+6.0), 5e-4) - 1
                else:
                    raise ValueError("Unknown redshift range to use for "
                        "template type {}".format(self._rrtype))
                zmin = self._redshifts[0]
                zmax = self._redshifts[-1]
                print("DEBUG: Using default redshift range {:.4f}-{:.4f} for "
                    "{}".format(zmin, zmax, os.path.basename(filename)))
            else:
                zmin = self._redshifts[0]
                zmax = self._redshifts[-1]
                print("DEBUG: Using redshift range {:.4f}-{:.4f} for "
                    "{}".format(zmin, zmax, os.path.basename(filename)))

            self._subtype = None
            if 'RRSUBTYP' in hdr:
                self._subtype = hdr['RRSUBTYP'].strip().upper()
            else:
                self._subtype = ''

        else:
            self._rrtype = spectype
            self._redshifts = redshifts
            self.wave = wave
            self.flux = flux
            self._subtype = subtype

        self._nbasis = self.flux.shape[0]
        self._nwave = self.flux.shape[1]


    @property
    def nbasis(self):
        return self._nbasis

    @property
    def nwave(self):
        return self._nwave

    @property
    def template_type(self):
        return self._rrtype

    @property
    def sub_type(self):
        return self._subtype

    @property
    def full_type(self):
        """Return formatted type:subtype string.
        """
        if self._subtype != '':
            return '{}:::{}'.format(self._rrtype, self._subtype)
        else:
            return self._rrtype

    @property
    def redshifts(self):
        return self._redshifts


    def eval(self, coeff, wave, z):
        """Return template for given coefficients, wavelengths, and redshift

        Args:
            coeff : array of coefficients length self.nbasis
            wave : wavelengths at which to evaluate template flux
            z : redshift at which to evaluate template flux

        Returns:
            template flux array

        Notes:
            A single factor of (1+z)^-1 is applied to the resampled flux
            to conserve integrated flux after redshifting.

        """
        assert len(coeff) == self.nbasis
        flux = self.flux.T.dot(coeff).T / (1+z)
        return trapz_rebin(self.wave*(1+z), flux, wave)




def find_templates(template_dir=None):
    """Return list of redrock-\*.fits template files

    Search directories in this order, returning results from first one found:
        - template_dir
        - $RR_TEMPLATE_DIR
        - <redrock_code>/templates/

    Args:
        template_dir (str): optional directory containing the templates.

    Returns:
        list: a list of template files.

    """
    if template_dir is None:
        if 'RR_TEMPLATE_DIR' in os.environ:
            template_dir = os.environ['RR_TEMPLATE_DIR']
        else:
            thisdir = os.path.dirname(__file__)
            tempdir = os.path.join(os.path.abspath(thisdir), 'templates')
            if os.path.exists(tempdir):
                template_dir = tempdir

    if template_dir is None:
        raise IOError("ERROR: can't find template_dir, $RR_TEMPLATE_DIR, or {rrcode}/templates/")
    else:
        print('DEBUG: Read templates from {}'.format(template_dir) )

    return sorted(glob(os.path.join(template_dir, 'rrtemplate-*.fits')))


class DistTemplatePiece(object):
    """One piece of the distributed template data.

    This is a simple container for storing interpolated templates for a set of
    redshift values.  It is used for communicating the interpolated templates
    between processes.

    In the MPI case, each process will store at most two of these
    simultaneously.  This is the data that is computed on a single process and
    passed between processes.

    Args:
        index (int): the chunk index of this piece- this corresponds to
            the process rank that originally computed this piece.
        redshifts (array): the redshift range contained in this piece.
        data (list): a list of dictionaries, one for each redshift, and
            each containing the 2D interpolated template values for all
            "wavehash" keys.

    """
    def __init__(self, index, redshifts, data):
        self.index = index
        self.redshifts = redshifts
        self.data = data


def _mp_rebin_template(template, dwave, zlist, qout):
    """Function for multiprocessing version of rebinning.
    """
    try:
        results = dict()
        for z in zlist:
            binned = rebin_template(template, z, dwave)
            results[z] = binned
        qout.put(results)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "MP rebin: {}".format(x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()
    return


class DistTemplate(object):
    """Distributed template data interpolated to all redshifts.

    For a given template, the redshifts are distributed among the
    processes in the communicator.  Then each process will rebin the
    template to those redshifts for the wavelength grids specified by
    dwave.

    Args:
        template (Template): the template to distribute
        dwave (dict): the keys are the "wavehash" and the values
            are a 1D array containing the wavelength grid.
        mp_procs (int): if not using MPI, restrict the number of
            multiprocesses to this.
        comm (mpi4py.MPI.Comm): (optional) the MPI communicator.

    """
    def __init__(self, template, dwave, mp_procs=1, comm=None):
        self._comm = comm
        self._template = template
        self._dwave = dwave

        self._comm_rank = 0
        self._comm_size = 1
        if self._comm is not None:
            self._comm_rank = self._comm.rank
            self._comm_size = self._comm.size

        self._distredshifts = np.array_split(self._template.redshifts,
            self._comm_size)

        myz = self._distredshifts[self._comm_rank]
        nz = len(myz)

        data = list()

        # In the case of not using MPI (comm == None), one process is rebinning
        # all the templates.  In that scenario, use multiprocessing
        # workers to do the rebinning.

        if self._comm is not None:
            # MPI case- compute our local redshifts
            for z in myz:
                binned = rebin_template(self._template, z, self._dwave)
                data.append(binned)
        else:
            # We don't have MPI, so use multiprocessing
            import multiprocessing as mp

            qout = mp.Queue()
            work = np.array_split(myz, mp_procs)
            procs = list()
            for i in range(mp_procs):
                p = mp.Process(target=_mp_rebin_template,
                    args=(self._template, self._dwave, work[i], qout))
                procs.append(p)
                p.start()

            # Extract the output into a single list
            results = dict()
            for i in range(mp_procs):
                res = qout.get()
                results.update(res)
            for z in myz:
                data.append(results[z])

        # Correct spectra for Lyman-series
        for i, z in enumerate(myz):
            for k in list(self._dwave.keys()):
                T = transmission_Lyman(z,self._dwave[k])
                for vect in range(data[i][k].shape[1]):
                    data[i][k][:,vect] *= T

        self._piece = DistTemplatePiece(self._comm_rank, myz, data)


    @property
    def comm(self):
        return self._comm

    @property
    def template(self):
        return self._template

    @property
    def local(self):
        return self._piece


    def cycle(self):
        """Pass our piece of data to the next process.

        If we have returned to our original data, then return True, otherwise
        return False.

        Args:
            Nothing

        Returns (bool):
            Whether we have finished (True) else False.

        """
        # If we are not using MPI, this function is a no-op, so just return.
        if self._comm is None:
            return True

        rank = self._comm_rank
        nproc = self._comm_size

        to_proc = rank + 1
        if to_proc >= nproc:
            to_proc = 0

        from_proc = rank - 1
        if from_proc < 0:
            from_proc = nproc - 1

        # Send our data and get a request handle for later checking.

        req = self._comm.isend(self._piece, to_proc)

        # Receive our data

        incoming = self._comm.recv(source=from_proc)

        # Wait for send to finishself._comm_rank = self._comm.rank

        req.wait()

        # Now replace our local piece with the new one

        self._piece = incoming

        # Are we done?

        done = False
        if self._piece.index == rank:
            done = True

        return done


def load_dist_templates(dwave, templates=None, comm=None, mp_procs=1):
    """Read and distribute templates from disk.

    This reads one or more template files from disk and distributes them among
    an MPI communicator.  Each process will locally store interpolated data
    for a redshift slice of each template.  For a single redshift, the template
    is interpolated to the wavelength grids specified by "dwave".

    As an example, imagine 3 templates with independent redshift ranges.  Also
    imagine that the communicator has 2 processes.  This function would return
    a list of 3 DistTemplate objects.  Within each of those objects, the 2
    processes store the interpolated data for a subset of the redshift range:

    DistTemplate #1:  zmin1 <---- p0 ----> | <---- p1 ----> zmax1
    DistTemplate #2:  zmin2 <-- p0 --> | <-- p1 --> zmax2
    DistTemplate #3:  zmin3 <--- p0 ---> | <--- p1 ---> zmax3

    Args:
        dwave (dict): the dictionary of wavelength grids.  Keys are the
            "wavehash" and values are an array of wavelengths.
        templates (str or None): if None, find all templates from the
            redrock template directory.  If a path to a file is specified,
            load that single template.  If a path to a directory is given,
            load all templates in that directory.
        comm (mpi4py.MPI.Comm): (optional) the MPI communicator.
        mp_procs (int): if not using MPI, restrict the number of
            multiprocesses to this.

    Returns:
        list: a list of DistTemplate objects.

    """
    timer = elapsed(None, "", comm=comm)

    template_files = None

    if (comm is None) or (comm.rank == 0):
        # Only one process needs to do this
        if templates is not None:
            if os.path.isfile(templates):
                # we are using just a single file
                template_files = [ templates ]
            elif os.path.isdir(templates):
                # this is a template dir
                template_files = find_templates(template_dir=templates)
            else:
                print("{} is neither a file nor a directory"\
                    .format(templates))
                sys.stdout.flush()
                if comm is not None:
                    comm.Abort()
        else:
            template_files = find_templates()

    if comm is not None:
        template_files = comm.bcast(template_files, root=0)

    template_data = list()
    if (comm is None) or (comm.rank == 0):
        for t in template_files:
            template_data.append(Template(filename=t))

    if comm is not None:
        template_data = comm.bcast(template_data, root=0)

    timer = elapsed(timer, "Read and broadcast of {} templates"\
        .format(len(template_files)), comm=comm)

    # Compute the interpolated templates in a distributed way with every
    # process generating a slice of the redshift range.

    dtemplates = list()
    for t in template_data:
        dtemplates.append(DistTemplate(t, dwave, mp_procs=mp_procs, comm=comm))

    timer = elapsed(timer, "Rebinning templates", comm=comm)

    return dtemplates


def eval_model(data, wave, R=None, templates=None):
    """Evaluate model spectra.

    Given a bunch of fits with coefficients COEFF, redshifts Z, and types
    SPECTYPE, SUBTYPE in data, evaluate the redrock model fits at the
    wavelengths wave using resolution matrix R.

    The wavelength and resolution matrices may be dictionaries including for
    multiple cameras.

    Args:
        data (array, [nspec]): array containing information on each model to
            evaluate.  Must contain at least Z, COEFF, SPECTYPE, and SUBTYPE
            fields.
        wave (array [nwave] or dictionary thereof): array of wavelengths in
            angstrom at which to evaluate the models.
        R (list of [nwave, nwave] arrays of floats or dictionary thereof):
            resolution matrices for evaluating spectra.
        templates (dictionary of Template): dictionary with (SPECTYPE, SUBTYPE)
            giving the template corresponding to each type.

    Returns:
        model fluxes, array [nspec, nwave].  If wave and R are dictionaries, then
        a dictionary of model fluxes, one for each camera.
    """
    if templates is None:
        templates = dict()
        templatefn = find_templates()
        for fn in templatefn:
            tx = Template(fn)
            templates[(tx.template_type, tx.sub_type)] = tx
    if isinstance(wave, dict):
        Rdict = R if R is not None else {x: None for x in wave}
        return {x: eval_model(data, wave[x], R=Rdict[x], templates=templates)
                for x in wave}
    out = np.zeros((len(data), len(wave)), dtype='f4')
    for i in range(len(data)):
        tx = templates[(data['SPECTYPE'][i], data['SUBTYPE'][i])]
        coeff = data['COEFF'][i][0:tx.nbasis]
        model = tx.flux.T.dot(coeff).T
        mx = trapz_rebin(tx.wave*(1+data['Z'][i]), model, wave)
        if R is None:
            out[i] = mx
        else:
            out[i] = R[i].dot(mx)
    return out
