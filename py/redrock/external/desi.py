'''
redrock wrapper tools for DESI
'''
from __future__ import absolute_import, division, print_function

import os, sys
import re
import warnings
from collections import OrderedDict

if sys.version_info[0] > 2:
    basestring = str

import numpy as np
import scipy.sparse

from astropy.io import fits
from astropy.table import Table

from desiutil.io import encode_table

import desispec.io
from desispec.resolution import Resolution

from ..sharedmem_mpi import MPIShared

from ..dataobj import (Target, MultiprocessingSharedSpectrum, 
    SimpleSpectrum, MPISharedTargets)

from .. import io
from .. import zfind


def write_zbest(outfile, zbest):
    '''
    Write zbest Table to outfile

    Adds blank BRICKNAME and SUBTYPE columns if needed
    Adds zbest.meta['EXTNAME'] = 'ZBEST'
    '''
    ntargets = len(zbest)
    if 'BRICKNAME' not in zbest.colnames:
        zbest['BRICKNAME'] = np.zeros(ntargets, dtype='S8')

    if 'SUBTYPE' not in zbest.colnames:
        zbest['SUBTYPE'] = np.zeros(ntargets, dtype='S8')

    zbest.meta['EXTNAME'] = 'ZBEST'
    zbest.write(outfile, format="fits", overwrite=True)


def read_spectra(spectrafiles, targetids=None, spectrum_class=SimpleSpectrum):
    '''
    Read targets from a list of spectra files.

    The spectrum_class argument is needed to use the same read_spectra routine
    for the two parallelism approaches for redrock. The multiprocessing version
    uses a shared memory at the spectrum class initialization, see the
    redrock.dataobj.MultiprocessingSharedSpectrum class, whereas the MPI
    version implements the shared memory after all spectra have been read by
    the root process, and so the MPI version used another more simple spectrum
    class (see redrock.dataobj.SimpleSpectrum).
   
    Args:
        spectrafiles : list of input spectra files, or string glob to match

    Options:
        targetids : list of target ids. If set, only those target spectra will
            be read.
        spectrum_class : the class to use for individual spectra.
    
    Returns tuple of (targets, meta) where
        targets is a list of Target objects and
        meta is a Table of metadata (currently only BRICKNAME)
    '''
    if isinstance(spectrafiles, basestring):
        import glob
        spectrafiles = glob.glob(spectrafiles)

    assert len(spectrafiles) > 0

    input_spectra = list()
    input_targetids = set()

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    for infile in spectrafiles:
        sp = desispec.io.read_spectra(infile)
        if hasattr(sp, 'fmap'):
            sp.fibermap = sp.fmap   #- for future compatibility
        input_spectra.append(sp)
        input_targetids.update(sp.fibermap['TARGETID'])

    if targetids is None:
        targetids = input_targetids

    targets = list()
    bricknames = list()
    for targetid in targetids:
        spectra = list()
        for sp in input_spectra:
            ii = (sp.fibermap['TARGETID'] == targetid)
            if np.count_nonzero(ii) == 0:
                continue
            if 'BRICKNAME' in sp.fibermap.dtype.names:
                brickname = sp.fibermap['BRICKNAME'][ii][0]
            else:
                brickname = 'unknown'
            for x in sp.bands:          #- typically 'b', 'r', 'z'
                wave = sp.wave[x]                
                flux = sp.flux[x][ii]
                ivar = sp.ivar[x][ii]*(sp.mask[x][ii]==0)
                Rdata = sp.resolution_data[x][ii]

                for i in range(flux.shape[0]):
                    if np.all(flux[i] == 0):
                        continue

                    if np.all(ivar[i] == 0):
                        continue

                    R = Resolution(Rdata[i])
                    spectra.append(spectrum_class(wave, flux[i], ivar[i], R))

        bricknames.append(brickname)
        #- end of for targetid in targetids loop

        if len(spectra) > 0:
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(\
                targetid, os.path.basename(brickfiles[0])))

    #- Create a metadata table in case we might want to add other columns
    #- in the future
    assert len(bricknames) == len(targets)
    dtype = [('BRICKNAME', 'S8'),]
    meta = np.zeros(len(bricknames), dtype=dtype)
    meta['BRICKNAME'] = bricknames

    return targets, meta


def read_bricks(brickfiles, trueflux=False, targetids=None, 
    spectrum_class=SimpleSpectrum):
    '''
    Read targets from a list of brickfiles

    The spectrum_class argument is needed to use the same read_spectra routine
    for the two parallelism approaches for redrock. The multiprocessing version
    uses a shared memory at the spectrum class initialization, see the
    redrock.dataobj.MultiprocessingSharedSpectrum class, whereas the MPI
    version implements the shared memory after all spectra have been read by
    the root process, and so the MPI version used another more simple spectrum
    class (see redrock.dataobj.SimpleSpectrum).

    Args:
        brickfiles : list of input brick files, or string glob to match
    
    Options:
        targetids : list of target ids. If set, only those target spectra will
            be read.
        spectrum_class : the class to use for individual spectra.
    
    Returns list of Target objects

    Note: these don't actually have to be bricks anymore; they are read via
        desispec.io.read_frame()
    '''
    if isinstance(brickfiles, basestring):
        import glob
        brickfiles = glob.glob(brickfiles)

    assert len(brickfiles) > 0

    bricks = list()
    brick_targetids = set()

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    for infile in brickfiles:
        b = desispec.io.read_frame(infile)
        bricks.append(b)
        brick_targetids.update(b.fibermap['TARGETID'])

    if targetids is None:
        targetids = brick_targetids

    targets = list()
    bricknames = list()
    for targetid in targetids:
        spectra = list()
        for brick in bricks:
            wave = brick.wave
            ii = (brick.fibermap['TARGETID'] == targetid)
            if np.count_nonzero(ii) == 0:
                continue

            if 'BRICKNAME' in brick.fibermap.dtype.names:
                brickname = brick.fibermap['BRICKNAME'][ii][0]
            else:
                brickname = 'unknown'
            flux = brick.flux[ii]
            ivar = brick.ivar[ii] * (brick.mask[ii]==0)
            Rdata = brick.resolution_data[ii]

            #- work around desispec.io.Brick returning 32-bit non-native endian
            # flux = flux.astype(float)
            # ivar = ivar.astype(float)
            # Rdata = Rdata.astype(float)

            for i in range(flux.shape[0]):
                if np.all(flux[i] == 0):
                    continue

                if np.all(ivar[i] == 0):
                    continue

                R = Resolution(Rdata[i])
                spectra.append(spectrum_class(wave, flux[i], ivar[i], R))

        #- end of for targetid in targetids loop

        if len(spectra) > 0:
            bricknames.append(brickname)
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(\
                targetid, os.path.basename(brickfiles[0])))

    #- Create a metadata table in case we might want to add other columns
    #- in the future
    assert len(bricknames) == len(targets)
    dtype = [('BRICKNAME', 'S8'),]
    meta = np.zeros(len(bricknames), dtype=dtype)
    meta['BRICKNAME'] = bricknames

    return targets, meta


class MPISharedTargetsDesi(MPISharedTargets):

    def __init__(self, comm, spectrafiles, targetids=None):
        """
        Collection of targets in MPI shared memory.

        This stores a collection of targets that are read from DESI spectra
        files in a buffered way.  Each HDU is stored unmodified in a separate
        shared memory buffer.  The target list is then generated as aliases
        into these shared memory locations.
        
        Args:
            comm (mpi4py.MPI.Comm): the communicator to use (or None).
            spectrafiles (list): the list of spectra files.
            targetids (list): (optional) the target IDs to select.
        """
        super().__init__(comm)

        # check the file list
        if isinstance(spectrafiles, basestring):
            import glob
            spectrafiles = glob.glob(spectrafiles)

        assert len(spectrafiles) > 0

        self._spectrafiles = spectrafiles

        # This is a dictionary (keyed on the filename) that stores the shared
        # HDU data.

        self._raw = OrderedDict()

        # This is the mapping between specs to targets for each file

        self._spec_to_target = {}
        self._target_specs = {}
        self._spec_keep = {}
        self._spec_sliced = {}

        # The bands for each file

        self._bands = {}
        self._band_map = {}

        # Resolution matrix properties

        self._res_shape = {}
        self._res_data_size = {}
        self._res_off_size = {}

        # The full list of targets from all files

        self._targetids = set()

        self._bricknames = {}

        # Loop over files

        for sfile in spectrafiles:
            self._raw[sfile] = OrderedDict()
            hdus = None
            nhdu = None
            fmap = None
            if self._root:
                hdus = fits.open(sfile, memmap=True)
                nhdu = len(hdus)
                fmap = encode_table(Table(hdus["FIBERMAP"].data, 
                    copy=True).as_array())
            if self._comm is not None:
                nhdu = self._comm.bcast(nhdu, root=0)
                fmap = self._comm.bcast(fmap, root=0)

            # Now every process has the fibermap and number of HDUs.  Build the
            # mapping between spectral rows and target IDs.

            keep_targetids = targetids
            if targetids is None:
                keep_targetids = fmap["TARGETID"]
            self._targetids.update(keep_targetids)

            if "BRICKNAME" in fmap.dtype.names:
                self._bricknames.update({ x : y for x,y in \
                    zip(fmap["TARGETID"], fmap["BRICKNAME"]) \
                    if x in keep_targetids })
            else:
                self._bricknames.update({ x : "unknown" for x in \
                    fmap["TARGETID"] if x in keep_targetids })

            # This is the spectral row to target mapping using the original
            # global indices (before slicing).

            self._spec_to_target[sfile] = [ x if y in keep_targetids else -1 \
                for x, y in enumerate(fmap["TARGETID"]) ]

            # The reduced set of spectral rows.

            self._spec_keep[sfile] = [ x for x in self._spec_to_target[sfile] \
                if x >= 0 ]

            # The mapping between original spectral indices and the sliced ones

            self._spec_sliced[sfile] = { x: y for y, x in \
                enumerate(self._spec_keep[sfile]) }

            # For each target, store the sliced row index of all spectra,
            # so that we can do a fast lookup later.

            self._target_specs[sfile] = {}
            for id in keep_targetids:
                self._target_specs[sfile][id] = [ x for x, y in \
                    enumerate(fmap["TARGETID"]) if y == id ]

            # Find the list of bands in this file

            self._bands[sfile] = []
            if self._root:
                for h in range(nhdu):
                    name = None
                    if "EXTNAME" not in hdus[h].header:
                        continue
                    name = hdus[h].header["EXTNAME"]
                    mat = re.match(r"(.*)_(.*)", name)
                    if mat is None:
                        continue
                    band = mat.group(1).lower()
                    if band not in self._bands[sfile]:
                        self._bands[sfile].append(band)
            if self._comm is not None:
                self._bands[sfile] = self._comm.bcast(self._bands[sfile], 
                    root=0)

            b = 0
            self._band_map[sfile] = {}
            for band in self._bands[sfile]:
                self._band_map[sfile][band] = b
                b += 1

            self._res_shape[sfile] = {}

            # Iterate over the data in HDU order and copy into the raw data
            # dictionary.  Only store spectra that match our list of targetids

            for h in range(nhdu):
                name = None
                if self._root:
                    if "EXTNAME" not in hdus[h].header:
                        name = "none"
                    else:
                        name = hdus[h].header["EXTNAME"]
                if self._comm is not None:
                    name = self._comm.bcast(name, root=0)
                # Find the band based on the name
                mat = re.match(r"(.*)_(.*)", name)
                if mat is None:
                    # This is the fibermap or some other HDU we don't recognize
                    continue
                band = mat.group(1).lower()
                type = mat.group(2)

                dims = None
                if self._root:
                    dims = hdus[h].data.shape
                    if len(dims) == 2 and dims[1] == 0:
                        dims = (dims[0],)
                if self._comm is not None:
                    dims = self._comm.bcast(dims, root=0)

                if type == "WAVELENGTH":
                    # This is a wavelength HDU, no need to slice it.
                    self._raw[sfile][name] = MPIShared(dims, 
                        np.dtype(np.float64), self._comm)
                    indata = None
                    if self._root:
                        indata = hdus[h].data.astype(np.float64)
                    self._raw[sfile][name].set(indata, (0,), 
                        fromrank=0)

                elif type == "RESOLUTION":
                    # This is a resolution matrix.  Instead of storing 
                    # the raw HDU data, we want to store scipy sparse matrix
                    # data so it is not duplicated later when we reconstruct 
                    # the spectra for the target list.
                    rows = self._spec_keep[sfile]
                    if np.count_nonzero(rows) == 0:
                        continue

                    # Root process is going to construct the resolution matrix
                    # for every spectrum, so that we can store those buffers
                    # in memory.  We will have separate buffers for all the
                    # "pieces" of the matrix data.

                    resshape = None
                    resdia_datasize = None
                    resdia_offsize = None
                    rescsr_datasize = None
                    rescsr_indxsize = None
                    rescsr_iptrsize = None

                    if self._root:
                        testdia = Resolution(hdus[h].data[rows[0]])
                        testcsr = testdia.tocsr()

                        resshape = testdia.shape
                        resdia_offsize = len(testdia.offsets.ravel())
                        resdia_datasize = len(testdia.data.ravel())
                        rescsr_indxsize = len(testcsr.indices)
                        rescsr_iptrsize = len(testcsr.indptr)
                        rescsr_datasize = len(testcsr.data)
                    if self._comm is not None:
                        resshape = self._comm.bcast(resshape, root=0)
                        resdia_offsize = self._comm.bcast(resdia_offsize, 
                            root=0)
                        resdia_datasize = self._comm.bcast(resdia_datasize, 
                            root=0)
                        rescsr_indxsize = self._comm.bcast(rescsr_indxsize, 
                            root=0)
                        rescsr_iptrsize = self._comm.bcast(rescsr_iptrsize, 
                            root=0)
                        rescsr_datasize = self._comm.bcast(rescsr_datasize, 
                            root=0)

                    self._res_shape[sfile][band] = resshape

                    res_diadata_name = \
                        "{}_RESOLUTION_DIA_DATA".format(band.upper())
                    res_diaoff_name = \
                        "{}_RESOLUTION_DIA_OFF".format(band.upper())
                    res_csrdata_name = \
                        "{}_RESOLUTION_CSR_DATA".format(band.upper())
                    res_csrindx_name = \
                        "{}_RESOLUTION_CSR_INDX".format(band.upper())
                    res_csriptr_name = \
                        "{}_RESOLUTION_CSR_IPTR".format(band.upper())

                    self._raw[sfile][res_diadata_name] = MPIShared(
                        (len(rows), resdia_datasize), np.dtype(np.float64),
                        self._comm)

                    self._raw[sfile][res_diaoff_name] = MPIShared(
                        (len(rows), resdia_offsize), np.dtype(np.int64),
                        self._comm)

                    self._raw[sfile][res_csrdata_name] = MPIShared(
                        (len(rows), rescsr_datasize), np.dtype(np.float64),
                        self._comm)

                    self._raw[sfile][res_csrindx_name] = MPIShared(
                        (len(rows), rescsr_indxsize), np.dtype(np.int64),
                        self._comm)

                    self._raw[sfile][res_csriptr_name] = MPIShared(
                        (len(rows), rescsr_iptrsize), np.dtype(np.int64),
                        self._comm)

                    for row in rows:
                        diadata = None
                        diaoff = None
                        csrdata = None
                        csrindx = None
                        csriptr = None
                        if self._root:
                            dia = Resolution(hdus[h].data[row])
                            csr = testdia.tocsr()

                            diadata = dia.data.astype(np.float64).reshape(1,-1)
                            diaoff = dia.offsets.astype(np.int64).reshape(1,-1)
                            csrdata = csr.data.astype(np.float64).reshape(1,-1)
                            csrindx = csr.indices.astype(np.int64)\
                                .reshape(1,-1)
                            csriptr = csr.indptr.astype(np.int64).reshape(1,-1)

                        self._raw[sfile][res_diadata_name].set(diadata, 
                            (row, 0), fromrank=0)

                        self._raw[sfile][res_diaoff_name].set(diaoff, 
                            (row, 0), fromrank=0)

                        self._raw[sfile][res_csrdata_name].set(csrdata, 
                            (row, 0), fromrank=0)

                        self._raw[sfile][res_csrindx_name].set(csrindx, 
                            (row, 0), fromrank=0)

                        self._raw[sfile][res_csriptr_name].set(csriptr, 
                            (row, 0), fromrank=0)

                else:
                    # This contains per-spectrum data.  Slice at the highest
                    # dimension to include only selected targets.
                    rows = self._spec_keep[sfile]
                    if np.count_nonzero(rows) == 0:
                        continue

                    self._raw[sfile][name] = MPIShared((len(rows), dims[1]),
                        np.dtype(np.float64), self._comm)

                    indata = None
                    if self._root:
                        indata = hdus[h].data[rows].astype(np.float64)
                    self._raw[sfile][name].set(indata, (0,0), 
                        fromrank=0)

        self._meta_dtype = [('BRICKNAME', 'S8'),]
        self._meta = np.zeros(len(self._bricknames), dtype=self._meta_dtype)
        self._meta['BRICKNAME'] = [ self._bricknames[x] for x \
            in sorted(self._targetids) ]


    @property
    def meta(self):
        return self._meta


    def _get_targets(self):
        """
        Generates a set of targets from the shared memory buffers.

        This is called by the base class targets().
        """
        targets = list()

        for id in sorted(self._targetids):
            speclist = list()
            for sfile in self._spectrafiles:
                if id not in self._target_specs[sfile]:
                    # nothing in this file
                    continue
                rows = self._target_specs[sfile][id]
                for row in rows:
                    memrow = self._spec_sliced[sfile][row]
                    for band in self._bands[sfile]:
                        wave_name = "{}_WAVELENGTH".format(band.upper())
                        flux_name = "{}_FLUX".format(band.upper())
                        ivar_name = "{}_IVAR".format(band.upper())
                        res_diadata_name = "{}_RESOLUTION_DIA_DATA".format(\
                            band.upper())
                        res_diaoff_name = "{}_RESOLUTION_DIA_OFF".format(\
                            band.upper())
                        res_csrdata_name = "{}_RESOLUTION_CSR_DATA".format(\
                            band.upper())
                        res_csrindx_name = "{}_RESOLUTION_CSR_INDX".format(\
                            band.upper())
                        res_csriptr_name = "{}_RESOLUTION_CSR_IPTR".format(\
                            band.upper())
                        wave_data = self._raw[sfile][wave_name][:]
                        flux_data = self._raw[sfile][flux_name][memrow][:]
                        ivar_data = None
                        if ivar_name in self._raw[sfile]:
                            ivar_data = self._raw[sfile][ivar_name][memrow][:]
                        resdia = None
                        rescsr = None
                        if res_diadata_name in self._raw[sfile]:
                            res_dia_off = self._raw[sfile][res_diaoff_name]\
                                [memrow]
                            res_dia_data = self._raw[sfile][res_diadata_name]\
                                [memrow].reshape((len(res_dia_off),-1))
                            resdia = scipy.sparse.dia_matrix((res_dia_data, 
                                res_dia_off), 
                                shape=self._res_shape[sfile][band], 
                                dtype=np.float64)
                            res_csr_indx = self._raw[sfile][res_csrindx_name]\
                                [memrow]
                            res_csr_iptr = self._raw[sfile][res_csriptr_name]\
                                [memrow]
                            res_csr_data = self._raw[sfile][res_csrdata_name]\
                                [memrow]
                            rescsr = scipy.sparse.csr_matrix((res_csr_data, 
                                res_csr_indx, res_csr_iptr), 
                                shape=self._res_shape[sfile][band], 
                                dtype=np.float64)
                        speclist.append( SimpleSpectrum(wave_data, flux_data,
                            ivar_data, resdia, Rcsr=rescsr) )

            # Create the Target from the list of spectra.  The
            # coadd is created on construction.
            targets.append(Target(id, speclist, coadd=None))

        return targets


def rrdesi(options=None, comm=None):
    import optparse
    from astropy.io import fits
    import time

    # Note, in the pure multiprocessing case (comm == None), "rank"
    # will always be set to zero, which is fine since we are outside
    # any areas using multiprocessing and this is just used to control
    # program flow.
    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    start_time = time.time()
    pid = os.getpid()
    
    parser = optparse.OptionParser(usage = \
        "%prog [options] spectra1 spectra2...")
    parser.add_option("-t", "--templates", type="string",
        help="template file or directory")
    parser.add_option("-o", "--output", type="string",
        help="output file")
    parser.add_option("--zbest", type="string",
        help="output zbest fits file")
    parser.add_option("-n", "--ntargets", type=int,
        help="number of targets to process")
    parser.add_option("--mintarget", type=int,
        help="first target to include", default=0)
    parser.add_option("--ncpu", type=int,
        help="number of cpu cores for multiprocessing", default=None)
    parser.add_option("--debug", help="debug with ipython", 
        action="store_true")
    parser.add_option("--allspec", 
        help="use individual spectra instead of coadd", action="store_true")

    if options is None:
        opts, infiles = parser.parse_args()
    else:
        opts, infiles = parser.parse_args(options)

    if (opts.output is None) and (opts.zbest is None):
        print('ERROR: --output or --zbest required')
        sys.exit(1)

    if len(infiles) == 0:
        print('ERROR: must provide input spectra/brick files')
        sys.exit(1)

    templates = None
    targets   = None
    meta      = None
    if rank == 0:
        t0 = time.time()
        print('INFO: reading targets')
        sys.stdout.flush()
        
        spectrum_class = SimpleSpectrum
        if opts.ncpu is None or opts.ncpu > 1:
            spectrum_class = MultiprocessingSharedSpectrum
        
        print('INFO: reading templates')
        sys.stdout.flush()
        
        if not opts.templates is None and os.path.isdir(opts.templates):
            templates = io.read_templates(template_list=None, template_dir=opts.templates)
        else:
            templates = io.read_templates(template_list=opts.templates, template_dir=None)

        dt = time.time() - t0
        
    # all processes get a copy of the templates from rank 0
    if comm is not None:
        templates = comm.bcast(templates, root=0)

    # Call zfind differently depending on our type of parallelism.

    meta = None
    if comm is not None:
        # Use MPI
        with MPISharedTargetsDesi(comm, infiles) as shared_targets:
            meta = shared_targets.meta
            zscan, zfit = zfind(shared_targets.targets, templates, 
            ncpu=None, comm=shared_targets.comm)
    else:
        # Use pure multiprocessing
        try:
            targets, meta = read_spectra(infiles, 
                spectrum_class=spectrum_class)
        except RuntimeError:
            targets, meta = read_bricks(infiles, 
                spectrum_class=spectrum_class)
            
        if not opts.allspec:
            for t in targets:
                t._all_spectra = t.spectra
                t.spectra = t.coadd

        if opts.ntargets is not None:
            targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
            meta = meta[opts.mintarget:opts.mintarget+opts.ntargets]

        zscan, zfit = zfind(targets, templates, ncpu=opts.ncpu)
    
    if rank == 0:
        if opts.output:
            print('INFO: writing {}'.format(opts.output))
            io.write_zscan(opts.output, zscan, zfit, clobber=True)

        if opts.zbest:
            zbest = zfit[zfit['znum'] == 0]

            #- Remove extra columns not needed for zbest
            zbest.remove_columns(['zz', 'zzchi2', 'znum'])

            #- Change to upper case like DESI
            for colname in zbest.colnames:
                if colname.islower():
                    zbest.rename_column(colname, colname.upper())

            #- Add brickname column
            zbest['BRICKNAME'] = meta['BRICKNAME']

            print('INFO: writing {}'.format(opts.zbest))
            write_zbest(opts.zbest, zbest)

    run_time = time.time() - start_time
    
    if comm is None or comm.rank == 0:
        print('INFO: finished {} in {:.1f} seconds'.format(\
            os.path.basename(infiles[0]), run_time))
    
    if opts.debug:
        if comm is not None:
            print('INFO: ignoring ipython debugging when using MPI')
        else:
            import IPython
            IPython.embed()

    return



