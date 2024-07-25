"""
redrock.templates
=================

Classes and functions for templates.
"""

from __future__ import absolute_import, division, print_function

import sys
from glob import glob
import os
import traceback

import numpy as np
from astropy.io import fits
from astropy.table import Table
from fitsio import FITS

from .utils import native_endian, elapsed
from .igm import transmission_Lyman
from .rebin import rebin_template, trapz_rebin
from .zscan import spectral_data

valid_template_methods = ('PCA', 'NMF')

class Template(object):
    """A spectral Template PCA object.

    The template data is read from a redrock-format template file.
    Alternatively, the data can be specified in the constructor.

    Args:
        filename (str): the path to the template file, either absolute or
            relative to the RR_TEMPLATE_DIR environment variable.

    """
    def __init__(self, filename=None, spectype=None, redshifts=None,
                 wave=None, flux=None, subtype=None, method='PCA',
                 igm_model='Inoue14',
                 zscan_galaxy=None, zscan_qso=None, zscan_star=None,
                 version=None):

        self._filename = filename
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

            if 'RRMETHOD' in hdr:
                self._method = hdr['RRMETHOD'].upper()
            else:
                self._method = 'PCA'

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
                    if zscan_galaxy is not None:
                        zmin, zmax, dz = zscan_galaxy.split(',')
                        self._redshifts = 10**np.arange(np.log10(1+float(zmin)), np.log10(1+float(zmax)), float(dz)) - 1
                    else:
                        self._redshifts = 10**np.arange(np.log10(1-0.005), np.log10(1+1.7), 3e-4) - 1
                elif self._rrtype == 'STAR':
                    if zscan_star is not None:
                        zmin, zmax, dz = zscan_star.split(',')
                        self._redshifts = np.arange(float(zmin), float(zmax), float(dz))
                    else:
                        self._redshifts = np.arange(-0.002, 0.00201, 4e-5)
                elif self._rrtype == 'QSO':
                    if zscan_qso is not None:
                        zmin, zmax, dz = zscan_qso.split(',')
                        self._redshifts = 10**np.arange(np.log10(1+float(zmin)), np.log10(1+float(zmax)), float(dz)) - 1
                    else:
                        self._redshifts = 10**np.arange(np.log10(1+0.05), np.log10(1+6.0), 5e-4) - 1
                else:
                    raise ValueError("Unknown redshift range to use for "
                        "template type {}".format(self._rrtype))
                zmin = self._redshifts[0]
                zmax = self._redshifts[-1]
                # print("DEBUG: Using default redshift range {:.4f}-{:.4f} for "
                #     "{}".format(zmin, zmax, os.path.basename(filename)))
            else:
                zmin = self._redshifts[0]
                zmax = self._redshifts[-1]
                # print("DEBUG: Using redshift range {:.4f}-{:.4f} for "
                #     "{}".format(zmin, zmax, os.path.basename(filename)))

            self._subtype = None
            if 'RRSUBTYP' in hdr:
                self._subtype = hdr['RRSUBTYP'].strip().upper()
            else:
                self._subtype = ''

            if 'RRIGM' in hdr:
                self._igm_model = hdr['RRIGM']
            else:
                #- auto-derive IGM model from known versions of pre-2024
                #- templates without RRIGM keyword
                if self._rrtype == 'STAR':
                    self._igm_model = 'None'
                elif self._rrtype == 'GALAXY' and self._version == '2.6':
                    # not actually needed for these galaxy templates that
                    # only go to z=1.7, but set anyway
                    self._igm_model = 'Calura12'
                elif self._rrtype == 'QSO' and self._version in ('0.1', '1.0'):
                    self._igm_model = 'Calura12'
                elif self._rrtype == 'QSO' and self._version == '1.1':
                    self._igm_model = 'Kamble20'
                else:
                    raise ValueError('Missing keyword RRIGM specifying IGM model to use')
        else:
            self._rrtype = spectype
            self._redshifts = redshifts
            self.wave = wave
            self.flux = flux
            self._subtype = subtype
            self._igm_model = igm_model
            self._method = method
            self._version = version

        self._nbasis = self.flux.shape[0]
        self._nwave = self.flux.shape[1]
        #It is much more efficient to copy wave and flux to GPU once
        #and store here rather than doing this every time, and keep track
        #of min and max wave as scalars on CPU
        self.minwave = self.wave[0]
        self.maxwave = self.wave[-1]
        self.gpuwave = None
        self.gpuflux = None

        if self._method not in valid_template_methods:
            raise ValueError(f'Template method {self._method} unrecognized; '
                              f'should be one of {valid_template_methods}')

        # Special case for GALAXY templates and PCA fitting, which can include
        # negative templates or coefficients: Cache a template that covers just
        # [OII].
        if self._rrtype == 'GALAXY' and self._method == "PCA":
            isOII = (3724 <= self.wave) & (self.wave <= 3733)
            self.OIItemplate = self.flux[:,isOII]

        if filename is not None:
            print(f'INFO: {os.path.basename(filename)} {self}')
        else:
            print(f'INFO: {self}')

    def __str__(self):
        zmin = round(self._redshifts[0], 4)
        zmax = round(self._redshifts[-1], 4)
        return f'{self._rrtype} {self._subtype} {self._method} IGM={self._igm_model} z={zmin}-{zmax}'

    def __repr__(self):
        return str(self)

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
        """Return formatted fulltype = spectype:::subtype string.
        """
        return make_fulltype(self._rrtype, self._subtype)

    @property
    def version(self):
        return self._version

    @property
    def filename(self):
        return self._filename

    @property
    def redshifts(self):
        return self._redshifts

    @property
    def solve_matrices_algorithm(self):
        """Return a string representing the algorithm to be used in
        zscan.solve_matrices.  Possible values are:
        PCA
        NMF
        Logic can be added here to select a default algorithm based on header
        keywords etc so that different templates seamlessly use different
        algorithms.
        """
        return self._method

    @property
    def method(self):
        """
        Alias for self.solve_matrices_algorithm, i.e. PCA or NMF
        """
        return self._method

    @property
    def igm_model(self):
        return self._igm_model


    def eval(self, coeff, wave, z, R=None):
        """Return template for given coefficients, wavelengths, and redshift

        Args:
            coeff : array of coefficients length self.nbasis
            wave : wavelengths at which to evaluate template flux
            z : redshift at which to evaluate template flux

        Options:
            R : array[nwave,nwave] resolution matrix to convolve with model

        Returns:
            template flux array

        Notes:
            No factors of (1+z) are applied to the resampled flux, i.e.
            evaluating the same coeffs at different z does not conserve
            integrated flux, but more directly maps model=templates.dot(coeff)
        """
        coeff = coeff[0:self.nbasis]
        flux = self.flux.T.dot(coeff).T
        model = trapz_rebin(self.wave*(1+z), flux, wave)
        if R is not None:
            model = R.dot(model)

        return model

def parse_fulltype(fulltype):
    """Parse template fulltype into (spectype, subtype)
    """
    fulltype = fulltype.strip()
    if ':::' in fulltype:
        spectype, subtype = fulltype.split(':::')
    else:
        spectype = fulltype
        subtype = ''

    return spectype, subtype

def make_fulltype(spectype, subtype):
    """combine (spectype, subtype) into fulltype
    """
    spectype = spectype.strip()
    if subtype is not None:
        subtype = subtype.strip()

    if subtype is not None and subtype != '':
        fulltype = f'{spectype}:::{subtype}'
    else:
        fulltype = spectype

    return fulltype

def find_templates(template_path=None):
    """Return list of Redrock template files

    `template_path` can be one of 4 things:

       * path to directory containing template files
       * path to single template file to use
       * path to text file listing which template files to use
       * None (use $RR_TEMPLATE_DIR instead)
    """
    if template_path is None:
        if 'RR_TEMPLATE_DIR' in os.environ:
            template_path = os.environ['RR_TEMPLATE_DIR']
        else:
            thisdir = os.path.dirname(__file__)
            tempdir = os.path.join(os.path.abspath(thisdir), 'templates')
            if os.path.exists(tempdir):
                template_path = tempdir

    if template_path is None:
        raise IOError("ERROR: can't find template_path, $RR_TEMPLATE_DIR, or {rrcode}/templates/")
    else:
        print(f'DEBUG: Reading templates from {template_path}')

    if os.path.isdir(template_path):
        default_templates_file = f'{template_path}/templates-default.txt'
        template_dir = template_path

    elif template_path.endswith('.txt'):
        if not os.path.exists(template_path):
            raise ValueError(f'Missing {template_path=}')
        default_templates_file = template_path
        template_dir = os.path.dirname(os.path.abspath(template_path))
    elif template_path.endswith( ('.fits', '.fits.gz', '.fits.fz') ):
        #- single template file, return that as a list
        return [template_path,]
    else:
        raise ValueError(f'Unrecognized {template_path=}')

    if os.path.exists(default_templates_file):
        #- New style (Jan 2024): default_templates.txt says which to use
        template_files = list()
        with open(default_templates_file) as fx:
            for line in fx.readlines():
                #- Strip comment lines and blank lines
                line = line.strip()
                if len(line) < 2 or line.startswith('#'):
                    continue
                else:
                    filename = line.split()[0] #- allow trailing comments
                    #- support absolute path and relative paths
                    if not line.startswith('/'):
                        filename = f'{template_dir}/{filename}'

                    if os.path.exists(filename):
                        template_files.append(filename)
                    else:
                        raise ValueError(f'missing {filename} given in {default_templates_file}')
    else:
        #- Old style: use all templates found in template directory
        template_files = sorted(glob(os.path.join(template_dir, 'rrtemplate-*.fits')))

    return template_files


def header2templatefiles(hdr, template_dir=None):
    """Derive template filenames from header keywords

    Args:
        hdr: dict-like with keys TEMNAMnn+TEMVERnn and/or TEMFILnn

    Options:
        template_dir (str): full path to redrock-templates directory

    Returns list of template filenames, including directory path
    """
    filenames = list()
    for i in range(100):
        filekey = f'TEMFIL{i:02d}'
        typekey = f'TEMNAM{i:02d}'
        verkey = f'TEMVER{i:02d}'
        if filekey in hdr:
            filenames.append( hdr[filekey] )
        elif (typekey in hdr) and (verkey in hdr):
            version = hdr[verkey].strip()
            spectype, subtype = parse_fulltype(hdr[typekey])

            #- special case unversioned STAR templates
            if version=='unknown':
                if spectype == 'STAR':
                    version = '0.1'
                else:
                    raise ValueError(f'unknown version for {spectype=}')

            #- blank subtype -> None for filename purposes
            if subtype == '':
                subtype = None

            filenames.append(f'rrtemplate-{spectype}-{subtype}-v{version}.fits')

    #- preprend template directory
    if template_dir is None:
        template_dir = os.environ['RR_TEMPLATE_DIR']

    filenames = [os.path.join(template_dir, fn) for fn in filenames]

    return filenames


def load_templates_from_header(hdr, template_dir=None, asdict=False):
    """Return templates matching header keywords

    Args:
        hdr: dict-like with keys TEMNAMnn+TEMVERnn and/or TEMFILnn

    Options:
        template_dir (str): full path to redrock-templates directory
        asdict (bool): return dict keyed by (spectype, subtype) instead of list
    
    Returns list of Template objects
    """
    template_filenames = header2templatefiles(hdr, template_dir=template_dir)
    return load_templates(template_filenames, asdict=asdict)

def load_templates(template_path=None, zscan_galaxy=None, zscan_star=None, zscan_qso=None,
                   asdict=False):
    """
    Return list or dict of Template objects

    Options:
        template_path: list of template files, directory, or text file with list of templates
        zscan_galaxy (str): zmin,zmax,dz redshift range for galaxies
        zscan_star (str): zmin,zmax,dz redshift range for stars
        zscan_qso (str): zmin,zmax,dz redshift range for QSOs
        asdict (bool): return dict keyed by (spectype, subtype) instead of list

    `template_path` is list of template file paths, or path to provide to
    find_templates, i.e. a path to a directory with templates, a path to
    a text file containing a list of templates, a path to a single template
    file, or None to use $RR_TEMPLATE_DIR instead.

    Returns: list or dict of Template objects

    Note: this always returns a list/dict, even if template_path is a path to a
    single template file.
    """
    if isinstance(template_path, (str, type(None))):
        template_path = find_templates(template_path)

    print(f'Reading templates from {template_path}')

    templates = list()
    for filename in template_path:
        tx = Template(filename, zscan_galaxy=zscan_galaxy,
                      zscan_star=zscan_star, zscan_qso=zscan_qso)
        templates.append(tx)

    if asdict:
        templates_dict = dict()
        for tx in templates:
            templates_dict[(tx.template_type, tx.sub_type)] = tx

        return templates_dict
    else:
        return templates


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


def _mp_rebin_template(template, dwave, zlist, qout, iproc, use_gpu):
    """Function for multiprocessing version of rebinning.
    With rebinning now done in batch mode, use process index, iproc
    to keep track of order of redshifts instead of keying dict by individual
    redshifts.
    """
    try:
        #New signature and return type for rebin_template 8/16/22 CW
        results = rebin_template(template, zlist, dwave, use_gpu=use_gpu)
        #Wrap in dict keyed by process index so redshifts can be
        #reassembled in correct order
        qout.put({ iproc: results })
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
        use_gpu (bool): (optional) If this process uses GPU
        gpu_mode (bool): (optional) If ANY process uses GPU - the reason
            we need both is that in GPU mode, the rebinning of all redshifts
            is done on all GPUs (redundantly but much faster than using
            MPI allgather) and no redshifts are done on any CPU only proc.

    """
    def __init__(self, template, dwave, mp_procs=1, comm=None, use_gpu=False, gpu_mode=False):
        self._comm = comm
        self._template = template
        self._dwave = dwave

        self._comm_rank = 0
        self._comm_size = 1
        if self._comm is not None:
            self._comm_rank = self._comm.rank
            self._comm_size = self._comm.size

        if (gpu_mode):
            #If ANY process is in GPU mode, distribute all redshifts to every
            #GPU process and no redshifts to any CPU process
            if (use_gpu):
                myz = self._template.redshifts
            else:
                #Create empty _piece
                myz = np.array([])
                data = dict()
                self._piece = DistTemplatePiece(self._comm_rank, myz, data)
                return
        else:
            #Distribute among CPU processes
            self._distredshifts = np.array_split(self._template.redshifts,
                self._comm_size)
            myz = self._distredshifts[self._comm_rank]

        # In the case of not using MPI (comm == None), one process is rebinning
        # all the templates.  In that scenario, use multiprocessing
        # workers to do the rebinning.

        # Removed MPI vs multiprocessing branch - CW 2/8/23
        # faster to just call rebin_template on one proc without using mp.Queue
        # compute our local redshifts
        # This will rebin template for all z on either GPU or CPU and
        # return a dict of three 3-d arrays (nz x nlambda x nbasis)
        data = rebin_template(self._template, myz, self._dwave, use_gpu=use_gpu)

        # Correct spectra for Lyman-series
        for k in list(self._dwave.keys()):
            #New algorithm accepts all z as an array and returns T, a 2-d
            # matrix (nz, nlambda) as a cupy or numpy array
            T = transmission_Lyman(myz,self._dwave[k], use_gpu=use_gpu, always_return_array=False,
                                   model=template.igm_model)
            if (T is None):
                #Return value of None means that wavelenght regime
                #does not overlap Lyman transmission - continue here
                continue
            #Vectorize multiplication
            data[k] *= T[:,:,None]
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


class ReDistTemplate(DistTemplate):
    """Distributed template data interpolated to all redshifts.

    For a given template, the redshifts are distributed among the
    processes in the communicator.  Then each process will rebin the
    template to those redshifts for the wavelength grids specified by
    dwave. After rebinning, the full redshift ranges are redistributed to
    each process in the communicator.

    Args:
        template (Template): the template to distribute
        dwave (dict): the keys are the "wavehash" and the values
            are a 1D array containing the wavelength grid.
        mp_procs (int): if not using MPI, restrict the number of
            multiprocesses to this.
        comm (mpi4py.MPI.Comm): (optional) the MPI communicator.

    """
    def __init__(self, template, dwave, mp_procs=1, comm=None, use_gpu=False, gpu_mode=False):
        super().__init__(template, dwave, mp_procs=mp_procs, comm=comm, use_gpu=use_gpu, gpu_mode=gpu_mode)
        ### This class is now Deprecated as allgather is no longer used.
        # Each GPU process now rebins all z.
        return
        if comm is not None:
            data = [e for s in comm.allgather(self.local.data) for e in s]
            self._piece = DistTemplatePiece(0, self.template.redshifts, data)
        else:
            raise NotImplementedError("ReDistTemplate not implemented for non-MPI")

    def cycle(self):
        """This function is a no-op since redshift ranges have been redistributed.

        Args:
            Nothing

        Returns (bool):
            Always returns True

        """
        # assert len(self.local.redshifts) == len(self.template.redshifts)
        return True


def load_dist_templates(dwave, templates=None, comm=None, mp_procs=1,
                        zscan_galaxy=None, zscan_qso=None, zscan_star=None,
                        redistribute=False, use_gpu=False, gpu_mode=False):
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
        redistribute (bool): (optional) allgather rebinned templates
            after distributed rebinning so each process has the full
            redshift range for the template.

    Returns dist_templates: a list of DistTemplate objects, sampled at redshifts
    """
    timer = elapsed(None, "", comm=comm)

    template_dict = None
    if (comm is None) or (comm.rank == 0):
        template_dict = load_templates(templates, asdict=True,
                                       zscan_galaxy=zscan_galaxy,
                                       zscan_star=zscan_star,
                                       zscan_qso=zscan_qso)

    if comm is not None:
        template_dict = comm.bcast(template_dict, root=0)

    timer = elapsed(timer, "Read and broadcast of {} templates"\
        .format(len(template_dict)), comm=comm)

    if (use_gpu):
        import cupy as cp
        c = cp.ones(1)
    # Take this timer out of if (use_gpu) block - for reference it hangs
    # entire code if only some procs call it
    timer = elapsed(timer, "Creating GPU context", comm=comm)

    # Compute the interpolated templates in a distributed way with every
    # process generating a slice of the redshift range.
    dtemplates = list()
    for t in template_dict.values():
        if redistribute:
            dtemplate = ReDistTemplate(t, dwave, mp_procs=mp_procs, comm=comm, use_gpu=use_gpu, gpu_mode=gpu_mode)
        else:
            dtemplate = DistTemplate(t, dwave, mp_procs=mp_procs, comm=comm, use_gpu=use_gpu, gpu_mode=gpu_mode)
        dtemplates.append(dtemplate)

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
        data (table-like, [nspec]): table with information on each model to
            evaluate.  Must contain at least Z, COEFF, SPECTYPE, and SUBTYPE
            fields.
        wave (array [nwave] or dictionary thereof): array of wavelengths in
            angstrom at which to evaluate the models.
        R (list of [nwave, nwave] arrays of floats or dictionary thereof):
            resolution matrices for evaluating spectra.
        templates (dictionary of Template): dictionary with (SPECTYPE, SUBTYPE)
            giving the template corresponding to each type.

    Returns:
        model fluxes, array [nspec, nwave].  If wave and R are dict, then
        a dictionary of model fluxes, one for each camera.
    """
    if templates is None:
        templates = load_templates(asdict=True)

    #- if wave is a dictionary of wavelength arrays, recursively call
    #- eval_model to return dictionary of models
    if isinstance(wave, dict):
        if R is None:
            Rdict = {x: None for x in wave}
        else:
            if not isinstance(R, dict) or set(R.keys()) != set(wave.keys()):
                raise ValueError('R must be dict with same keys as wave')
            Rdict = R

        out = dict()
        for key in wave:
            out[key] = eval_model(data, wave[key], R=Rdict[key], templates=templates)

        return out

    models = np.zeros((len(data), len(wave)), dtype='f4')
    for i in range(len(data)):
        tx = templates[(data['SPECTYPE'][i], data['SUBTYPE'][i])]
        coeff = data['COEFF'][i][0:tx.nbasis]
        z = data['Z'][i]
        models[i] = tx.eval(coeff, wave, z, R=R[i])

    return models

