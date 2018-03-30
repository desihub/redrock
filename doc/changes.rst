==================
redrock Change Log
==================

0.10.2 (unreleased)
-------------------

* No changes yet.

0.10.1 (2018-03-30)
-------------------

* Default QSO redshift range 0.05-4.0 instead of 0.5-4.0 (PR `#107`_).

.. _`#107`: https://github.com/desihub/redrock/pull/104

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
