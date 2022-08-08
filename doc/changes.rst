==================
redrock Change Log
==================

0.16.0 (unreleased)
-------------------

* No changes yet.

0.16.0 (2022-08-08)
-------------------

* Add rrdesi ``if __name__ == "__main__"`` wrapper for multiprocessing
  robustness (PR `#209`_).
* Update GitHub test automation (PR `#210`_).
* Avoid bad fits when ranking zchi2 vs. z minima; fixes redshift pileup
  for new QSO templates at edges of zscan range (PR `#218`_).

.. _`#209`: https://github.com/desihub/redrock/pull/209
.. _`#210`: https://github.com/desihub/redrock/pull/210
.. _`#218`: https://github.com/desihub/redrock/pull/218

0.15.4 (2022-02-28)
-------------------

* Add redrock.templates.eval_model convenience routine (PR `#206`_).

.. _`#206`: https://github.com/desihub/redrock/pull/206

0.15.3 (2022-02-11)
-------------------

* Propagate SURVEY and PROGRAM keywords from input files (PR `#203`_).

.. _`#203`: https://github.com/desihub/redrock/pull/203

0.15.2 (2022-01-23)
-------------------

* Propagate spec group keywords from input files (PR `#202`_).

.. _`#202`: https://github.com/desihub/redrock/pull/202

0.15.1 (2022-01-20)
-------------------

* add dependency keywords to redrock output (PR `#200`_).
* set zwarn LITTLE_COVERAGE for badamp/badcol (PR `#201`_).

.. _`#200`: https://github.com/desihub/redrock/pull/200
.. _`#201`: https://github.com/desihub/redrock/pull/201

0.15.0 (2021-07-14)
-------------------

Note: Major changes to output formats; requires desispec >= 0.45.0

* Split FIBERMAP into FIBERMAP (coadded) and EXP_FIBERMAP (per-exposure)
  (PR `#196`_).
* Add additional ZWARN bit masking for known bad input data (PR `#196`_).
* Rename zbest -> redrock output, update rrdesi option names (PR `#198`_).

.. _`#196`: https://github.com/desihub/redrock/pull/196
.. _`#198`: https://github.com/desihub/redrock/pull/198

0.14.6 (2021-07-06)
-------------------

* reserve ZWARN bits 16-23 for end-user; redrock will not set these.
* Add tophap prior option (PR `#194`_).
* Switch to github actions for testing (PR `#195`_).

.. _`#194`: https://github.com/desihub/redrock/pull/194
.. _`#195`: https://github.com/desihub/redrock/pull/195

0.14.5 (2021-02-15)
-------------------

* Use temporary files + rename to avoid partially written files with the
  final name in case of timeout (PR `#186`_).

.. _`#186`: https://github.com/desihub/redrock/pull/186

0.14.4 (2020-08-03)
-------------------

* Re-enable ability for templates to specify their redshift range
  (one line update to master).

0.14.3 (2020-04-07)
-------------------

* Allow :func:`redrock.external.boss.read_spectra` to receive a
  string as well as a list of files (PR `#173`_).
* Support coadds that don't have EXPID in fibermap (master update).

.. _`#173`: https://github.com/desihub/redrock/pull/173


0.14.2 (2019-10-17)
-------------------

* Bug fix for specfiles of different sizes (PR `#167`_).
* Fix plotting subset of input spectra (PR `#168`_).
* Add `--no-mpi-abort` option (PR `#170`_)

.. _`#167`: https://github.com/desihub/redrock/pull/167
.. _`#168`: https://github.com/desihub/redrock/pull/168
.. _`#170`: https://github.com/desihub/redrock/pull/170

0.14.1 (2019-08-09)
-------------------

* Minor code cleanup (PRs `#162`_, `#164`_).
* Add `and_mask` option for BOSS (PR `#165`_).

.. _`#162`: https://github.com/desihub/redrock/pull/162
.. _`#164`: https://github.com/desihub/redrock/pull/164
.. _`#165`: https://github.com/desihub/redrock/pull/165

0.14.0 (2018-12-16)
-------------------

* Adds optional cosmic ray rejection during coadds (PR `#156`_).
* No longer requires BRICKNAME (PR `#157`_).
* Fix interactive plotspec window disappearing (PR `#161`_).

.. _`#156`: https://github.com/desihub/redrock/pull/156
.. _`#157`: https://github.com/desihub/redrock/pull/157
.. _`#161`: https://github.com/desihub/redrock/pull/161

0.13.2 (2018-11-07)
-------------------

Version used for 18.11 software release.

* Codacy style recommendations (PR `#155`_).
* Optional redshift prior (PR `#152`_).

.. _`#152`: https://github.com/desihub/redrock/pull/152
.. _`#155`: https://github.com/desihub/redrock/pull/155

0.13.1 (2018-09-26)
-------------------

* Fixed problem with new format of ``make_templates`` (PR `#153`_).
* Update code based on codacy recommendations (PR `#154`_).

.. _`#153`: https://github.com/desihub/redrock/pull/153
.. _`#154`: https://github.com/desihub/redrock/pull/154

0.13.0 (2018-08-31)
-------------------

* Lower galaxy z_min from +0.005 to -0.005 (PR `#136`_).
* Support for simutaneous fits of multiple e/BOSS spPlates (PR `#137`_,
  `#141`_, `#147`_).
* Bug fix when using subset of targetids (PR `#139`_).
* Small interface useability updates (PR `#142`_, `#143`_).
* Fix R normalization cut bug impacting tags 0.12.0 and 0.12.1 (PR `#144`_).
* Mask sky lines 5577 and 9793.5 (PR `#146`_).
* Standarize ZBEST output format for easier concatenating tables (PR `#149`_).

.. _`#136`: https://github.com/desihub/redrock/pull/136
.. _`#137`: https://github.com/desihub/redrock/pull/137
.. _`#139`: https://github.com/desihub/redrock/pull/139
.. _`#141`: https://github.com/desihub/redrock/pull/141
.. _`#142`: https://github.com/desihub/redrock/pull/142
.. _`#143`: https://github.com/desihub/redrock/pull/143
.. _`#144`: https://github.com/desihub/redrock/pull/144
.. _`#146`: https://github.com/desihub/redrock/pull/146
.. _`#147`: https://github.com/desihub/redrock/pull/147
.. _`#149`: https://github.com/desihub/redrock/pull/149

0.12.1 (2018-07-26)
-------------------

* Update DELTACHI2 column definition to match how it is used in ZWARN flag,
  i.e. excluding other candidates with nearby redshifts (PR `#134`_).

.. _`#134`: https://github.com/desihub/redrock/pull/134

0.12.0 (2018-07-18)
-------------------

* Adds optional archetypes (PR `#119`_).
* Include blank fibers in output with ZWARN NODATA flag (PR `#123`_).
* Include template name in output (PR `#124`_).
* Include template and archetype version numbers in zbest output
  (PR `#126`_, `#128`_, and `#131`_).
* Update travis testing to astropy=2 python=3 (PR `#127`_).
* Increase QSO redshift range to z=6 (PR `#130`_).
* rrplot option for a subset of targetids (PR `#132`_).

.. _`#119`: https://github.com/desihub/redrock/pull/119
.. _`#123`: https://github.com/desihub/redrock/pull/123
.. _`#124`: https://github.com/desihub/redrock/pull/124
.. _`#126`: https://github.com/desihub/redrock/pull/126
.. _`#127`: https://github.com/desihub/redrock/pull/127
.. _`#128`: https://github.com/desihub/redrock/pull/128
.. _`#130`: https://github.com/desihub/redrock/pull/130
.. _`#131`: https://github.com/desihub/redrock/pull/131
.. _`#132`: https://github.com/desihub/redrock/pull/132

0.11.0 (2018-05-10)
-------------------

* Catch LinAlgErrors from bad input data (PR `#109`_).
* Add --nminima option (PR `#113`_).
* Improve spectra reading speed (PR `#114`_).
* hdf5 file locking workaround (PR `#116`_).
* Fix MPI version of LyA transmission correction (PR `#117`_).
* WD DA and DB templates (PR `#118`_).

.. _`#109`: https://github.com/desihub/redrock/pull/109
.. _`#113`: https://github.com/desihub/redrock/pull/113
.. _`#114`: https://github.com/desihub/redrock/pull/114
.. _`#116`: https://github.com/desihub/redrock/pull/116
.. _`#117`: https://github.com/desihub/redrock/pull/117
.. _`#118`: https://github.com/desihub/redrock/pull/118

0.10.1 (2018-03-30)
-------------------

* Default QSO redshift range 0.05-4.0 instead of 0.5-4.0 (PR `#107`_).

.. _`#107`: https://github.com/desihub/redrock/pull/107

0.10.0 (2018-03-29)
-------------------

* Correct QSO template for LyA during zscan (PR `#104`_).

.. _`#104`: https://github.com/desihub/redrock/pull/104

0.9.0 (2018-02-23)
------------------

* ivar=0 for edge pix with integral(resolution)<0.99 (PR `#94`_)
* Restore --ncpu option (PR `#95`_)
* Adds wrap-redrock MPI wrapper script (PR `#97`_)
* Robust to input NaN and Inf (PR `#99`_)
* Adds WD templates (PR `#101`_)

.. _`#94`: https://github.com/desihub/redrock/pull/94
.. _`#95`: https://github.com/desihub/redrock/pull/95
.. _`#97`: https://github.com/desihub/redrock/pull/97
.. _`#99`: https://github.com/desihub/redrock/pull/99
.. _`#101`: https://github.com/desihub/redrock/pull/101

0.8.0 (2018-01-30)
------------------

* Major restructure of MPI and multiprocessing dataflow
  (PR `#67`_, `#73`_, `#76`_).
* Fully support desiInstall and DESI infrastructure generally (PR `#65`_).
* Fix import errors that were preventing RTD builds (PR `#91`_).
* Add seed to template generation; increase number of stars used (PR `#93`_).
* Add rrplot script to be called from ipython (PR `#90`_).

.. _`#65`: https://github.com/desihub/redrock/pull/65
.. _`#67`: https://github.com/desihub/redrock/pull/67
.. _`#73`: https://github.com/desihub/redrock/pull/73
.. _`#76`: https://github.com/desihub/redrock/pull/76
.. _`#90`: https://github.com/desihub/redrock/pull/90
.. _`#91`: https://github.com/desihub/redrock/pull/91
.. _`#93`: https://github.com/desihub/redrock/pull/93


0.7.0 (2017-12-20)
------------------

* no ZWARN SMALL_DELTA_CHI2 between same spectype (PR `#47`_)
* rrdesi --templates can now be folder not just file (PR `#44`_)
* Allow templates to optionally include redshift range (PR `#41`_)
* API CHANGE: redrock.io.read_templates() returns dict not list (PR `#41`_)
* set ivar = 0 where mask != 0 (PR `#42`_)
* Add NUMEXP and NUMTILE to zbest output (PR `#59`_)
* Propagate input fibermap into output zbest (PR `#59`_)

.. _`#47`: https://github.com/desihub/desispec/pull/47
.. _`#44`: https://github.com/desihub/desispec/pull/44
.. _`#41`: https://github.com/desihub/desispec/pull/41
.. _`#42`: https://github.com/desihub/desispec/pull/42
.. _`#59`: https://github.com/desihub/desispec/pull/59

0.6.0 (2017-11-10)
------------------

* adds rrboss to process boss spectra (PR `#37`_)
* refactors multiprocessing parallelism to use less memory (PR `#37`_)

.. _`#37`: https://github.com/desihub/desispec/pull/37

0.5.0 (2017-09-29)
------------------

* adds optional MPI parallelism (PR `#34`_)

.. _`#34`: https://github.com/desihub/desispec/pull/34

0.4.2 (2017-08-14)
------------------

* refactored multiprocessing parallelism to use explicit shared memory (PR `#31`_)

.. _`#31`: https://github.com/desihub/desispec/pull/31

0.4.1 (2017-06-16)
------------------

* add support for new DESI spectra format

0.4 (2017-02-03)
----------------

* add optional truth input to plotspec
* Fix bug when first target is missing a channel of data
* external.desi.read_bricks allow glob for list of brick files
* external.desi.read_bricks read subset of targetids from bricks
* add support for stars and template subtypes
* limit galaxy redshift scan to z<1.7

0.3 (2017-01-23)
----------------

* added this file
* python3 updates
* refactor internal data object wrappers
* fit and store multiple minima in chi2 vs. z
* refactor parallelism
* add option to fit coadd instead of individual spectra
* add plotspec
* experimental: penalize GALAXY template fits with negative [OII] flux

0.2 (2016-03-05)
----------------

* tag for DESI zdc1
