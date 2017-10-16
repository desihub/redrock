==================
Redrock change log
==================

0.5.1 (unreleased)
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
