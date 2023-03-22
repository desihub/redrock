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

Archetypes::

    If want to run with --nearest_nbh option, the user must clone archetypes as:
    git clone https://github.com/abhi0395/new-archetypes.git
    Then run rrdesi --archetypes <archetype_dir> --nearest_nbh
    Another required file is archetype file for galaxies that contain physical data and is stored at NERSC. If the rrdesi is run on NERSC, then the file would automatically be read. Otherwise there should be an io error. For help contact AbhijeetAnand@lbl.gov

Running
-------

To run on desi bricks files::

    rrdesi --zbest zbest.fits --output rrdetails.h5 brick*.fits

License
-------

redrock is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.

| Stephen Bailey & David Schlegel
| Lawrence Berkeley National Lab
| Spring 2017
