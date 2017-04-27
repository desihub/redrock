=======
redrock
=======

.. image:: https://travis-ci.org/sbailey/redrock.svg?branch=master
    :target: https://travis-ci.org/sbailey/redrock

Redshift fitting for spectroperfectionism.

To install:
```
git clone https://github.com/sbailey/redrock
cd redrock
git clone https://github.com/sbailey/redrock-templates py/redrock/templates
python setup.py install
```
That will install the templates with the code.  Alternatively, the templates
can be put elsewhere and set `$RR_TEMPLATE_DIR` to that location.

To run on desi bricks files:
```
rrdesi --zbest zbest.fits --output rrdetails.h5 brick*.fits
```

| Stephen Bailey & David Schlegel
| Lawrence Berkeley National Lab
| Spring 2017
