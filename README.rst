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

We recommend users see details of ``rrdesi`` before running ``redrock`` to understand the arguments in more detail.

Run::
    
    rrdesi --help

Running Redrock on desi spectra
-------------------------------

**1) Without Archetypes**::

    rrdesi -i <spectra_file> -o <output_file> -d <details_file.h5> 

**2) With Archetypes**
    
In archetype mode, redrock can be run either with one particular archetype file (e.g., GALAXY, QSO or STAR), or you can also provide the directory that contains archetypes for different spectral classes.

You can also define your own set of archetypes. However, users must follow the file structure of redrock archetypes. To start with, the user should clone the following repository to run redrock in Archetype mode.::

    git clone https://github.com/abhi0395/new-archetypes.git

Or::

    git clone https://github.com/desihub/redrock-archetypes.git

In summary, the archetypes method uses a combination of physical galaxy spectra and Legendre polynomials to construct a new set of templates and then solve for the coefficients using the bounded value least square method for a few redshifts defined by ``--nminima``. 

The method solves for the coefficients of the Legendre polynomials in each camera (b, r, z cameras of desi, ``--archetype-legendre-percamera`` keyword is introduced for that). Another argument is ``--archetype-legendre-prior``, which can be prescribed to add a prior while solving for the coefficients (e.g. ``--archetype-legendre-prior 0.1``). If ``--archetype-legendre-degree 0`` is provided, the method will only use archetypes to fit the spectra; no Legendre polynomials will be used. Note that a single ``--archetypes-no-legendre`` flag will deactivate all other archetype and legendre-related flags. 

If you do not want to use default values, you should separately provide those arguments without the ``--archetypes-no-legendre`` flag.

Example run (with all the default values, including per camera mode)::
    
    rrdesi -i <spectra_file> --archetypes <archetype_dir or archetype_file> -o <output_file> -d <details_file.h5> 

Example run (with all the default values but without per camera mode)::
    
    rrdesi -i <spectra_file> --archetypes <archetype_dir or archetype_file> -o <output_file> -d <details_file.h5> --archetype-per-camera False 

Example run (with no archetype related default values)::
    
    rrdesi -i <spectra_file> --archetypes <archetype_dir or archetype_file> -o <output_file> -d <details_file.h5> --archetypes-no-legendre

**3) Archetypes + Nearest neighbours (in chi2 space) approach**

Similar to archetypes (method - 2), it also looks for the nearest neighbours of the best archetypes in chi2 space, selects a few nearest neighbours (input provided by the user, ``--archetype-nnearest``) and then constructs a new set of templates combining these archetypes and Legendre polynomials to fit the galaxy spectra as described above. 

Example run::
        
    rrdesi -i <spectra_file> --archetypes <archetype_dir or archetype_file> -o <output_file> -d  <details_file.h5> --archetype-nnearest 2 

For comment or help regarding archetypes, please contact abhijeetanand2011_at_gmail_dot_com.


License
-------

redrock is free software licensed under a 3-clause BSD-style license. For details see
the ``LICENSE.rst`` file.

| Stephen Bailey & David Schlegel
| Lawrence Berkeley National Lab
| Spring 2017
