=======
redrock
=======

|Actions Status| |Coveralls Status| |Documentation Status|

.. |Actions Status| image:: https://github.com/desihub/redrock/workflows/CI/badge.svg
    :target: https://github.com/desihub/redrock/actions
    :alt: GitHub Actions CI Status

.. |Coveralls Status| image:: https://coveralls.io/repos/desihub/redrock/badge.svg
    :target: https://coveralls.io/github/desihub/redrock
    :alt: Test Coverage Status

.. |Documentation Status| image:: https://readthedocs.org/projects/redrock/badge/?version=latest
    :target: https://redrock.readthedocs.io/en/latest/
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
