"""
redrock.external.desi
=====================

redrock wrapper tools for DESI
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import re
import warnings
import traceback

import argparse

if sys.version_info[0] > 2:
    basestring = str

import numpy as np

from astropy.io import fits
from astropy.table import Table

from desiutil.io import encode_table

from desispec.resolution import Resolution

from ..utils import elapsed, get_mp, distribute_work

from ..targets import (Spectrum, Target, DistTargets)

from ..templates import load_dist_templates

from ..results import write_zscan

from ..zfind import zfind

from .._version import __version__

from ..archetypes import All_archetypes


def write_zbest(outfile, zbest, fibermap, template_version, archetype_version):
    """Write zbest and fibermap Tables to outfile

    Args:
        outfile (str): output path.
        zbest (Table): best fit table.
        fibermap (Table): the fibermap from the original inputs.

    """
    header = fits.Header()
    header['RRVER'] = (__version__, 'Redrock version')
    for i, fulltype in enumerate(template_version.keys()):
        header['TEMNAM'+str(i).zfill(2)] = fulltype
        header['TEMVER'+str(i).zfill(2)] = template_version[fulltype]
    if not archetype_version is None:
        for i, fulltype in enumerate(archetype_version.keys()):
            header['ARCNAM'+str(i).zfill(2)] = fulltype
            header['ARCVER'+str(i).zfill(2)] = archetype_version[fulltype]
    zbest.meta['EXTNAME'] = 'ZBEST'
    fibermap.meta['EXTNAME'] = 'FIBERMAP'

    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU(header=header))
    hx.append(fits.convenience.table_to_hdu(zbest))
    hx.append(fits.convenience.table_to_hdu(fibermap))
    hx.writeto(outfile, overwrite=True)
    return


class DistTargetsDESI(DistTargets):
    """Distributed targets for DESI.

    DESI spectral data is grouped by sky location, but is just a random
    collection of spectra for all targets.  Reading this into memory with
    spectra grouped by target ID involves not just loading the data, but
    also sorting it by target.

    We pass through the spectra files once to compute all the book-keeping
    associated with regrouping the spectra by target.  Then we pass through
    again and actually read and distribute the data.

    Args:
        spectrafiles (str or list): a list of input files or pattern match
            of files.
        coadd (bool): if False, do not compute the coadds.
        targetids (list): (optional) restrict the global set of target IDs
            to this list.
        first_target (int): (optional) integer offset of the first target to
            consider in each file.  Useful for debugging / testing.
        n_target (int): (optional) number of targets to consider in each file.
            Useful for debugging / testing.
        comm (mpi4py.MPI.Comm): (optional) the MPI communicator.
        cache_Rcsr: pre-calculate and cache sparse CSR format of resolution
            matrix R
    """

    ### @profile
    def __init__(self, spectrafiles, coadd=True, targetids=None,
        first_target=None, n_target=None, comm=None, cache_Rcsr=False):

        comm_size = 1
        comm_rank = 0
        if comm is not None:
            comm_size = comm.size
            comm_rank = comm.rank

        # check the file list
        if isinstance(spectrafiles, basestring):
            import glob
            spectrafiles = glob.glob(spectrafiles)

        assert len(spectrafiles) > 0

        self._spectrafiles = spectrafiles

        # This is the mapping between specs to targets for each file

        self._spec_to_target = {}
        self._target_specs = {}
        self._spec_keep = {}
        self._spec_sliced = {}

        # The bands for each file

        self._bands = {}
        self._wave = {}

        # The full list of targets from all files

        self._alltargetids = set()

        # The fibermaps from all files

        self._fmaps = {}

        for sfile in spectrafiles:
            hdus = None
            nhdu = None
            fmap = None
            if comm_rank == 0:
                hdus = fits.open(sfile, memmap=True)
                nhdu = len(hdus)
                fmap = encode_table(Table(hdus["FIBERMAP"].data,
                    copy=True).as_array())

            if comm is not None:
                nhdu = comm.bcast(nhdu, root=0)
                fmap = comm.bcast(fmap, root=0)

            # Now every process has the fibermap and number of HDUs.  Build the
            # mapping between spectral rows and target IDs.

            if targetids is None:
                keep_targetids = sorted(fmap["TARGETID"])
            else:
                keep_targetids = sorted(targetids)

            # Select a subset of the target range from each file if desired.

            if first_target is None:
                first_target = 0
            if first_target > len(keep_targetids):
                raise RuntimeError("first_target value \"{}\" is beyond the "
                    "number of selected targets in the file".\
                    format(first_target))

            if n_target is None:
                n_target = len(keep_targetids)
            if first_target + n_target > len(keep_targetids):
                raise RuntimeError("Requested first_target / n_target "
                    " range is larger than the number of selected targets "
                    " in the file")

            keep_targetids = keep_targetids[first_target:first_target+n_target]

            self._alltargetids.update(keep_targetids)

            # This is the spectral row to target mapping using the original
            # global indices (before slicing).

            self._spec_to_target[sfile] = [ x if y in keep_targetids else -1 \
                for x, y in enumerate(fmap["TARGETID"]) ]

            # The reduced set of spectral rows.

            self._spec_keep[sfile] = [ x for x in self._spec_to_target[sfile] \
                if x >= 0 ]

            # The mapping between original spectral indices and the sliced ones

            self._spec_sliced[sfile] = { x : y for y, x in \
                enumerate(self._spec_keep[sfile]) }

            # Slice the fibermap

            self._fmaps[sfile] = fmap[self._spec_keep[sfile]]

            # For each target, store the sliced row index of all spectra,
            # so that we can do a fast lookup later.

            self._target_specs[sfile] = {}
            for id in keep_targetids:
                self._target_specs[sfile][id] = [ x for x, y in \
                    enumerate(self._fmaps[sfile]["TARGETID"]) if y == id ]

            # We need some more metadata information for each file-
            # specifically, the bands that are used and their wavelength grids.
            # That information will allow us to pre-allocate our local target
            # list and then fill that with one pass through all HDUs in the
            # files.

            self._bands[sfile] = []
            self._wave[sfile] = dict()

            if comm_rank == 0:
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
                    htype = mat.group(2)

                    if htype == "WAVELENGTH":
                        self._wave[sfile][band] = \
                            hdus[h].data.astype(np.float64).copy()

            if comm is not None:
                self._bands[sfile] = comm.bcast(self._bands[sfile], root=0)
                self._wave[sfile] = comm.bcast(self._wave[sfile], root=0)

            if comm_rank == 0:
                hdus.close()

        self._keep_targets = list(sorted(self._alltargetids))

        # Now we have the metadata for all targets in all files.  Distribute
        # the targets among process weighted by the amount of work to do for
        # each target.  This weight is either "1" if we are going to use coadds
        # or the number of spectra if we are using all the data.

        tweights = None
        if not coadd:
            tweights = dict()
            for t in self._keep_targets:
                tweights[t] = 0
                for sfile in spectrafiles:
                    if t in self._target_specs[sfile]:
                        tweights[t] += len(self._target_specs[sfile][t])

        self._proc_targets = distribute_work(comm_size,
            self._keep_targets, weights=tweights)

        self._my_targets = self._proc_targets[comm_rank]

        # Reverse mapping- target ID to index in our list
        self._my_target_indx = {y : x for x, y in enumerate(self._my_targets)}

        # Now every process has its local target IDs assigned.  Pre-create our
        # local target list with empty spectral data (except for wavelengths)

        self._my_data = list()

        for t in self._my_targets:
            speclist = list()
            tileids = set()
            exps = set()
            bname = None
            for sfile in spectrafiles:
                hastileid = ("TILEID" in self._fmaps[sfile].colnames)
                for b in self._bands[sfile]:
                    if t in self._target_specs[sfile]:
                        nspec = len(self._target_specs[sfile][t])
                        for s in range(nspec):
                            sindx = self._target_specs[sfile][t][s]
                            frow = self._fmaps[sfile][sindx]
                            if bname is None:
                                bname = frow["BRICKNAME"]
                            exps.add(frow["EXPID"])
                            if hastileid:
                                tileids.add(frow["TILEID"])
                            speclist.append(Spectrum(self._wave[sfile][b],
                                None, None, None, None))
            # Meta dictionary for this target.  Whatever keys we put in here
            # will end up as columns in the final zbest output table.
            tmeta = dict()
            tmeta["NUMEXP"] = len(exps)
            tmeta["NUMEXP_datatype"] = "i4"
            tmeta["NUMTILE"] = len(tileids)
            tmeta["NUMTILE_datatype"] = "i4"
            tmeta["BRICKNAME"] = bname
            tmeta["BRICKNAME_datatype"] = "S8"
            self._my_data.append(Target(t, speclist, coadd=False, meta=tmeta))

        # Iterate over the data and broadcast.  Every process selects the rows
        # of each table that contain pieces of local target data and copies it
        # into place.

        # these are for tracking offsets within the spectra for each target.
        tspec_flux = { x : 0 for x in self._my_targets }
        tspec_ivar = tspec_flux.copy()
        tspec_mask = tspec_flux.copy()
        tspec_res = tspec_flux.copy()

        for sfile in spectrafiles:
            rows = self._spec_keep[sfile]
            if len(rows) == 0:
                continue

            hdus = None
            if comm_rank == 0:
                hdus = fits.open(sfile, memmap=True)

            for b in self._bands[sfile]:
                extname = "{}_{}".format(b.upper(), "FLUX")
                hdata = None
                badflux = None
                if comm_rank == 0:
                    hdata = hdus[extname].data[rows]
                    # check for NaN and Inf here (should never happen of course)
                    badflux = np.isnan(hdata) | np.isinf(hdata) | np.isneginf(hdata)
                    hdata[badflux] = 0.0
                if comm is not None:
                    hdata = comm.bcast(hdata, root=0)
                    badflux = comm.bcast(badflux, root=0)
                    
                toff = 0
                for t in self._my_targets:
                    if t in self._target_specs[sfile]:
                        for trow in self._target_specs[sfile][t]:
                            self._my_data[toff].spectra[tspec_flux[t]].flux = \
                                hdata[trow].astype(np.float64).copy()
                            tspec_flux[t] += 1
                    toff += 1

                extname = "{}_{}".format(b.upper(), "IVAR")
                hdata = None
                if comm_rank == 0:
                    hdata = hdus[extname].data[rows]
                    # check for NaN and Inf here (should never happen of course)
                    bad = np.isnan(hdata) | np.isinf(hdata) | np.isneginf(hdata)
                    hdata[bad] = 0.0
                    hdata[badflux] = 0.0 # also set ivar=0 to bad flux                    
                if comm is not None:
                    hdata = comm.bcast(hdata, root=0)

                toff = 0
                for t in self._my_targets:
                    if t in self._target_specs[sfile]:
                        for trow in self._target_specs[sfile][t]:
                            self._my_data[toff].spectra[tspec_ivar[t]].ivar = \
                                hdata[trow].astype(np.float64).copy()
                            tspec_ivar[t] += 1
                    toff += 1
                
                extname = "{}_{}".format(b.upper(), "MASK")
                hdata = None
                if comm_rank == 0:
                    if extname in hdus:
                        hdata = hdus[extname].data[rows]
                if comm is not None:
                    hdata = comm.bcast(hdata, root=0)

                if hdata is not None:
                    toff = 0
                    for t in self._my_targets:
                        if t in self._target_specs[sfile]:
                            for trow in self._target_specs[sfile][t]:
                                self._my_data[toff].spectra[tspec_mask[t]]\
                                    .ivar *= (hdata[trow] == 0)
                                tspec_mask[t] += 1
                        toff += 1

                extname = "{}_{}".format(b.upper(), "RESOLUTION")
                hdata = None
                if comm_rank == 0:
                    hdata = hdus[extname].data[rows]

                if comm is not None:
                    hdata = comm.bcast(hdata, root=0)

                toff = 0
                for t in self._my_targets:
                    if t in self._target_specs[sfile]:
                        for trow in self._target_specs[sfile][t]:
                            dia = Resolution(hdata[trow].astype(np.float64))
                            self._my_data[toff].spectra[tspec_res[t]].R = dia
                            #- Coadds replace Rcsr so only compute if not coadding
                            if not coadd and cache_Rcsr:
                                self._my_data[toff].spectra[tspec_res[t]].Rcsr = dia.tocsr()
                            tspec_res[t] += 1
                    toff += 1

                del hdata

            if comm_rank == 0:
                hdus.close()

        # Compute the coadds now if we are going to use those

        if coadd:
            for t in self._my_data:
                t.compute_coadd(cache_Rcsr)

        self.fibermap = Table(np.hstack([ self._fmaps[x] \
            for x in self._spectrafiles ]))

        super(DistTargetsDESI, self).__init__(self._keep_targets, comm=comm)


    def _local_target_ids(self):
        return self._my_targets

    def _local_data(self):
        return self._my_data


def rrdesi(options=None, comm=None):
    """Estimate redshifts for DESI targets.

    This loads distributed DESI targets from one or more spectra grouping
    files and computes the redshifts.  The outputs are written to a redrock
    scan file and a DESI redshift catalog.

    Args:
        options (list): optional list of commandline options to parse.
        comm (mpi4py.Comm): MPI communicator to use.

    """
    global_start = elapsed(None, "", comm=comm)

    parser = argparse.ArgumentParser(description="Estimate redshifts from"
        " DESI target spectra.")

    parser.add_argument("-t", "--templates", type=str, default=None,
        required=False, help="template file or directory")

    parser.add_argument("--archetypes", type=str, default=None,
        required=False, help="archetype file or directory for final redshift comparison")

    parser.add_argument("-o", "--output", type=str, default=None,
        required=False, help="output file")

    parser.add_argument("-z", "--zbest", type=str, default=None,
        required=False, help="output zbest FITS file")

    parser.add_argument("--targetids", type=str, default=None,
        required=False, help="comma-separated list of target IDs")

    parser.add_argument("--mintarget", type=int, default=None,
        required=False, help="first target to process in each file")

    parser.add_argument("--priors", type=str, default=None,
        required=False, help="optional redshift prior file")

    parser.add_argument("-n", "--ntargets", type=int,
        required=False, help="the number of targets to process in each file")

    parser.add_argument("--nminima", type=int, default=3,
        required=False, help="the number of redshift minima to search")

    parser.add_argument("--allspec", default=False, action="store_true",
        required=False, help="use individual spectra instead of coadd")

    parser.add_argument("--ncpu", type=int, default=None,
        required=False, help="DEPRECATED: the number of multiprocessing"
            " processes; use --mp instead")

    parser.add_argument("--mp", type=int, default=0,
        required=False, help="if not using MPI, the number of multiprocessing"
            " processes to use (defaults to half of the hardware threads)")

    parser.add_argument("--no-skymask", default=False, action="store_true",
        required=False, help="Do not do extra masking of sky lines")

    parser.add_argument("--debug", default=False, action="store_true",
        required=False, help="debug with ipython (only if communicator has a "
        "single process)")

    parser.add_argument("infiles", nargs='*')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.ncpu is not None:
        print('WARNING: --ncpu is deprecated; use --mp instead')
        args.mp = args.ncpu

    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.size
        comm_rank = comm.rank

    # Check arguments- all processes have this, so just check on the first
    # process

    if comm_rank == 0:
        if args.debug and comm_size != 1:
            print("--debug can only be used if the communicator has one "
                " process")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

        if (args.output is None) and (args.zbest is None):
            parser.print_help()
            print("ERROR: --output or --zbest required")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()
            else:
                sys.exit(1)

        if len(args.infiles) == 0:
            print("ERROR: must provide input files")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()
            else:
                sys.exit(1)

        if (args.targetids is not None) and ((args.mintarget is not None) \
            or (args.ntargets is not None)):
            print("ERROR: cannot select targets by both ID and range")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()
            else:
                sys.exit(1)

    targetids = None
    if args.targetids is not None:
        targetids = [ int(x) for x in args.targetids.split(",") ]

    n_target = None
    if args.ntargets is not None:
        n_target = args.ntargets

    first_target = None
    if args.mintarget is not None:
        first_target = args.mintarget
    elif n_target is not None:
        first_target = 0

    # Multiprocessing processes to use if MPI is disabled.
    mpprocs = 0
    if comm is None:
        mpprocs = get_mp(args.mp)
        print("Running with {} processes".format(mpprocs))
        if "OMP_NUM_THREADS" in os.environ:
            nthread = int(os.environ["OMP_NUM_THREADS"])
            if nthread != 1:
                print("WARNING:  {} multiprocesses running, each with "
                    "{} threads ({} total)".format(mpprocs, nthread,
                    mpprocs*nthread))
                print("WARNING:  Please ensure this is <= the number of "
                    "physical cores on the system")
        else:
            print("WARNING:  using multiprocessing, but the OMP_NUM_THREADS")
            print("WARNING:  environment variable is not set- your system may")
            print("WARNING:  be oversubscribed.")
        sys.stdout.flush()
    elif comm_rank == 0:
        print("Running with {} processes".format(comm_size))
        sys.stdout.flush()

    try:
        # Load and distribute the targets
        if comm_rank == 0:
            print("Loading targets...")
            sys.stdout.flush()

        start = elapsed(None, "", comm=comm)

        # Load the targets.  If comm is None, then the target data will be
        # stored in shared memory.
        targets = DistTargetsDESI(args.infiles, coadd=(not args.allspec),
            targetids=targetids, first_target=first_target, n_target=n_target,
            comm=comm, cache_Rcsr=True)

        #- Mask some problematic sky lines
        if not args.no_skymask:
            for t in targets.local():
                for s in t.spectra:
                    ii = (5572. <= s.wave) & (s.wave <= 5582.)
                    ii |= (9792. <= s.wave) & (s.wave <= 9795.)
                    s.ivar[ii] = 0.0

        # Get the dictionary of wavelength grids
        dwave = targets.wavegrids()

        stop = elapsed(start, "Read and distribution of {} targets"\
            .format(len(targets.all_target_ids)), comm=comm)

        # Read the template data

        dtemplates = load_dist_templates(dwave, templates=args.templates,
            comm=comm, mp_procs=mpprocs)

        # Compute the redshifts, including both the coarse scan and the
        # refinement.  This function only returns data on the rank 0 process.

        start = elapsed(None, "", comm=comm)

        scandata, zfit = zfind(targets, dtemplates, mpprocs,
            nminima=args.nminima, archetypes=args.archetypes, priors=args.priors)

        stop = elapsed(start, "Computing redshifts took", comm=comm)

        # Write the outputs

        if args.output is not None:
            start = elapsed(None, "", comm=comm)
            if comm_rank == 0:
                write_zscan(args.output, scandata, zfit, clobber=True)
            stop = elapsed(start, "Writing zscan data took", comm=comm)

        if args.zbest:
            start = elapsed(None, "", comm=comm)
            if comm_rank == 0:
                zbest = zfit[zfit['znum'] == 0]

                # Remove extra columns not needed for zbest
                zbest.remove_columns(['zz', 'zzchi2', 'znum'])

                # Change to upper case like DESI
                for colname in zbest.colnames:
                    if colname.islower():
                        zbest.rename_column(colname, colname.upper())

                template_version = {t._template.full_type:t._template._version for t in dtemplates}
                archetype_version = None
                if not args.archetypes is None:
                    archetypes = All_archetypes(archetypes_dir=args.archetypes).archetypes
                    archetype_version = {name:arch._version for name, arch in archetypes.items() }
                write_zbest(args.zbest, zbest, targets.fibermap, template_version, archetype_version)

            stop = elapsed(start, "Writing zbest data took", comm=comm)

    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "Proc {}: {}".format(comm_rank, x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()
        if comm is not None:
            comm.Abort()

    global_stop = elapsed(global_start, "Total run time", comm=comm)

    if args.debug:
        import IPython
        IPython.embed()

    return
