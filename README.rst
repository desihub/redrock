=======
redrock
=======

.. image:: https://travis-ci.org/desihub/redrock.svg?branch=master
    :target: https://travis-ci.org/desihub/redrock
    :alt: Travis Build Status

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
