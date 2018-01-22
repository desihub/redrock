==================
redrock Change Log
==================

0.7.1 (unreleased)
------------------

* Fully support desiInstall and DESI infrastructure generally (PR `#65`_).
* Fix import errors that were preventing builds.

.. _`#65`: https://github.com/desihub/redrock/pull/65

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
