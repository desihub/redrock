#!/bin/bash -x

#- Fail early, fail often
set -e

# CONDA
conda create --yes -n test -c astropy-ci-extras python=$PYTHON_VERSION pip
source activate test

# CORE DEPENDENCIES
# conda install --yes pytest Cython jinja2 psutil pyyaml requests

# NUMPY scipy
conda install --yes numpy scipy astropy h5py numba

# Get templates

cd py/redrock
RR_TEMPLATE_VER=0.2
wget https://github.com/sbailey/redrock-templates/archive/${RR_TEMPLATE_VER}.tar.gz
tar -xzf ${RR_TEMPLATE_VER}.tar.gz
mv redrock-templates-${RR_TEMPLATE_VER} templates
rm ${RR_TEMPLATE_VER}.tar.gz
