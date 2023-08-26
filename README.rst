=======
redrock
=======

.. image:: https://travis-ci.org/desihub/redrock.svg?branch=master
    :target: https://travis-ci.org/desihub/redrock
    :alt: Travis Build Status

.. image:: https://coveralls.io/repos/github/desihub/redrock/badge.svg?branch=master
    :target: https://coveralls.io/github/desihub/redrock?branch=master
    :alt: Test Coverage Status

.. image:: https://readthedocs.org/projects/redrock/badge/?version=latest
    :target: http://redrock.readthedocs.org/en/latest/
    :alt: Documentation Status

Introduction
------------

Redshift fitting for spectroperfectionism.

Installation
------------

To install::

    git clone https://github.com/desihub/redrock
    cd redrock
    git clone https://github.com/desihub/redrock-templates py/redrock/templates
    python setup.py install

That will install the templates with the code.  Alternatively, the templates
can be put elsewhere and set ``$RR_TEMPLATE_DIR`` to that location.

Users are recommended to see details of `rrdesi` before running `redrock` to understand the      arguments in more details.

Run::
    
    rrdesi --help

Running Redrock on desi spectra
-------------------------------

**1) Without Archetypes**::

    rrdesi -i <spectra_file> --output <output_file> --details <details_file.h5> -n 1

**2) With Archetypes**::
    
In archetype mode, redrock can be run either with one particular archetype file (e.g., GALAXY, QSO or STAR) or you can also provide the full directory that contains archetypes for all spectypes.

You can also define your own set of archetypes, however users must follow the file structure of redrock-archetypes. To start with, user should clone following repository to run redrock in Archetype mode::
    
    git clone https://github.com/abhi0395/new-archetypes.git

Or::

    git clone https://github.com/desihub/redrock-archetypes.git

Example run::
    
    rrdesi -i <spectra_file> --archetypes <archetype_dir or archetype_file> --output <output_file> --details <details_file.h5> -deg_legendre 2 --nminima 9

**3) Archetypes + Nearest neighbours (in chi2 space) approach**::

Similar to archetypes (method - 2) but also looks for the nearest neighbours of the bestarchetypes in chi2 space. Then uses a combination of those nearest neighbours and Legendre polynomials to fit the galaxy spectra using bounded value least square method.

Example run ::
        
    rrdesi -i <spectra_file> --archetypes <archetype_dir or archetype_file> -o <output_file> -d  <details_file.h5> -deg_legendre 2 -n_nearest 2

For comment or help regarding archetypes please contact AbhijeetAnand [at] lbl.gov


License
-------

redrock is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.

| Stephen Bailey & David Schlegel
| Lawrence Berkeley National Lab
| Spring 2017
