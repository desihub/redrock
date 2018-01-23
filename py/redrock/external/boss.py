"""
redrock.external.boss
=====================

redrock wrapper tools for BOSS
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import re
import warnings
import traceback

import argparse

import numpy as np
from scipy import sparse

from astropy.io import fits
from astropy.table import Table

import fitsio

import desispec.resolution
from desispec.resolution import Resolution

from ..utils import elapsed, get_mp, distribute_work

from ..targets import Spectrum, Target, DistTargetsCopy

from ..templates import load_dist_templates

from ..results import write_zscan

from ..zfind import zfind


def platemjdfiber2targetid(plate, mjd, fiber):
    return plate*1000000000 + mjd*10000 + fiber


def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return (plate, mjd, fiber)


def write_zbest(outfile, zbest):
    """Write zbest Table to outfile

    Args:
        outfile (str): output file.
        zbest (Table): the output best fit results.

    """
    zbest.meta['EXTNAME'] = 'ZBEST'

    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU())
    hx.append(fits.convenience.table_to_hdu(zbest))
    hx.writeto(outfile, overwrite=True)
    return


def read_spectra(spplate_name, targetids=None, use_frames=False,
    fiberid=None, coadd=False):
    """Read targets from a list of spectra files

    Args:
        spplate_name (str): input spPlate file
        targetids (list): restrict targets to this subset.
        use_frames (bool): if True, use frames.
        fiberid (int): Use this fiber ID.
        coadd (bool): if True, compute and use the coadds.

    Returns:
        tuple: (targets, meta) where targets is a list of Target objects and
        meta is a Table of metadata (currently only BRICKNAME).

    """
    ## read spplate
    spplate = fitsio.FITS(spplate_name)
    plate = spplate[0].read_header()["PLATEID"]
    mjd = spplate[0].read_header()["MJD"]
    if not use_frames:
        infiles = [spplate_name]
    if use_frames:
        path = os.path.dirname(spplate_name)
        cameras = ['b1','r1','b2','r2']

        infiles = []
        nexp_tot=0
        for c in cameras:
            try:
                nexp = spplate[0].read_header()["NEXP_{}".format(c.upper())]
            except ValueError:
                print("DEBUG: spplate {} has no exposures in camera {} ".format(spplate_name,c))
                continue
            for i in range(1,nexp+1):
                nexp_tot += 1
                expid = str(nexp_tot)
                if nexp_tot<10:
                    expid = '0'+expid
                exp = path+"/spCFrame-"+spplate[0].read_header()["EXPID"+expid][:11]+".fits"
                infiles.append(exp)

    spplate.close()
    bricknames={}
    dic_spectra = {}

    for infile in infiles:
        h = fitsio.FITS(infile)
        assert plate == h[0].read_header()["PLATEID"]
        fs = h[5]["FIBERID"][:]
        if fiberid is not None:
            w = np.in1d(fs,fiberid)
            fs = fs[w]

        fl = h[0].read()
        iv = h[1].read()*(h[2].read()==0)
        wd = h[4].read()

        ## crop to lmin, lmax
        lmin = 3500.
        lmax = 10000.
        if use_frames:
            la = 10**h[3].read()
            if h[0].read_header()["CAMERAS"][0]=="b":
                lmax = 6000.
            else:
                lmin = 5500.
        else:
            coeff0 = h[0].read_header()["COEFF0"]
            coeff1 = h[0].read_header()["COEFF1"]
            la = 10**(coeff0 + coeff1*np.arange(fl.shape[1]))
            la = np.broadcast_to(la,fl.shape)

        imin = abs(la-lmin).min(axis=0).argmin()
        imax = abs(la-lmax).min(axis=0).argmin()

        la = la[:,imin:imax]
        fl = fl[:,imin:imax]
        iv = iv[:,imin:imax]
        wd = wd[:,imin:imax]

        w = wd<1e-5
        wd[w]=2.
        ii = np.arange(la.shape[1])
        di = ii-ii[:,None]
        di2 = di**2
        ndiag = int(4*np.ceil(wd.max())+1)
        nbins = wd.shape[1]

        for f in fs:
            i = (f-1)
            if use_frames:
                i = i%500
            if np.all(iv[i]==0):
                print("DEBUG: skipping plate,fid = {},{} (no data)".format(plate,f))
                continue

            t = platemjdfiber2targetid(plate, mjd, f)
            if t not in dic_spectra:
                dic_spectra[t]=[]
                brickname = '{}-{}'.format(plate,mjd)
                bricknames[t] = brickname

            ## build resolution from wdisp
            reso = np.zeros([ndiag,nbins])
            for idiag in range(ndiag):
                offset = ndiag//2-idiag
                d = np.diagonal(di2,offset=offset)
                if offset<0:
                    reso[idiag,:len(d)] = np.exp(-d/2/wd[i,:len(d)]**2)
                else:
                    reso[idiag,nbins-len(d):nbins]=np.exp(-d/2/wd[i,nbins-len(d):nbins]**2)

            R = Resolution(reso)
            ccd = sparse.spdiags(1./R.sum(axis=1).T, 0, *R.shape)
            R = (ccd*R).todia()
            dic_spectra[t].append(Spectrum(la[i], fl[i], iv[i], R, R.tocsr()))

        h.close()
        print("DEBUG: read {} ".format(infile))

    if targetids == None:
        targetids = sorted(list(dic_spectra.keys()))

    targets = []
    for targetid in targetids:
        spectra = dic_spectra[targetid]
        # Add the brickname to the meta dictionary.  The keys of this dictionary
        # will end up as extra columns in the output ZBEST HDU.
        tmeta = dict()
        tmeta["BRICKNAME"] = bricknames[targetid]
        tmeta["BRICKNAME_datatype"] = "S8"
        if len(spectra) > 0:
            targets.append(Target(targetid, spectra, coadd=coadd, meta=tmeta))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(targetid, os.path.basename(brickfiles[0])))

    #- Create a metadata table in case we might want to add other columns
    #- in the future
    assert len(bricknames.keys()) == len(targets)

    metatable = Table(names=("TARGETID", "BRICKNAME"), dtype=("i8", "S8",))
    for t in targetids:
        metatable.add_row( (t, bricknames[t]) )

    return targets, metatable


def rrboss(options=None, comm=None):
    """Estimate redshifts for BOSS targets.

    This loads targets serially and copies them into a DistTargets class.
    It then runs redshift fitting and writes the output to a catalog.

    Args:
        options (list): optional list of commandline options to parse.
        comm (mpi4py.Comm): MPI communicator to use.

    """
    global_start = elapsed(None, "", comm=comm)

    parser = argparse.ArgumentParser(description="Estimate redshifts from"
        " BOSS target spectra.")

    parser.add_argument("--spplate", type=str, default=None,
        required=True, help="input plate directory")

    parser.add_argument("-t", "--templates", type=str, default=None,
        required=False, help="template file or directory")

    parser.add_argument("-o", "--output", type=str, default=None,
        required=False, help="output file")

    parser.add_argument("--zbest", type=str, default=None,
        required=False, help="output zbest FITS file")

    parser.add_argument("--targetids", type=str, default=None,
        required=False, help="comma-separated list of target IDs")

    parser.add_argument("--mintarget", type=int, default=None,
        required=False, help="first target to process")

    parser.add_argument("-n", "--ntargets", type=int,
        required=False, help="the number of targets to process")

    parser.add_argument("--nminima", type=int, default=3,
        required=False, help="the number of redshift minima to search")

    parser.add_argument("--allspec", default=False, action="store_true",
        required=False, help="use individual spectra instead of coadd")

    parser.add_argument("--mp", type=int, default=0,
        required=False, help="if not using MPI, the number of multiprocessing"
            " processes to use (defaults to half of the hardware threads)")

    parser.add_argument("--use-frames", default=False, action="store_true",
        required=False, help="use individual spcframes instead of spplate "
        "(the spCFrame files are expected to be in the same directory as "
        "the spPlate")

    parser.add_argument("--broadband-degree", type=int, default=None,
        required=False, help="Degree of broadband parameters (default is no broadbands)")

    parser.add_argument("--debug", default=False, action="store_true",
        required=False, help="debug with ipython (only if communicator has a "
        "single process)")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

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
            print("--output or --zbest required")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

        if (args.targetids is not None) and ((args.mintarget is not None) \
            or (args.ntargets is not None)):
            print("cannot select targets by both ID and range")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

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

        # Read the spectra on the root process.  Currently the "meta" Table
        # returned here is not propagated to the output zbest file.  However,
        # that could be changed to work like the DESI write_zbest() function.
        # Each target contains metadata which is propagated to the output zbest
        # table though.
        targets, meta = read_spectra(args.spplate, targetids=targetids,
            use_frames=args.use_frames, coadd=(not args.allspec))

        if args.ntargets is not None:
            targets = targets[first_target:first_target+n_targets]
            meta = meta[first_target:first_target+n_targets]

        stop = elapsed(start, "Read of {} targets"\
            .format(len(targets)), comm=comm)

        # Distribute the targets.

        start = elapsed(None, "", comm=comm)

        dtargets = DistTargetsCopy(targets, comm=comm, root=0)

        # Get the dictionary of wavelength grids
        dwave = dtargets.wavegrids()

        stop = elapsed(start, "Distribution of {} targets"\
            .format(len(dtargets.all_target_ids)), comm=comm)

        # Read the template data

        dtemplates = load_dist_templates(dwave, templates=args.templates,
            comm=comm, mp_procs=mpprocs, bb_deg=args.broadband_degree)

        # Compute the redshifts, including both the coarse scan and the
        # refinement.  This function only returns data on the rank 0 process.

        start = elapsed(None, "", comm=comm)

        scandata, zfit = zfind(dtargets, dtemplates, mpprocs,
            nminima=args.nminima)

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

                write_zbest(args.zbest, zbest)

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
