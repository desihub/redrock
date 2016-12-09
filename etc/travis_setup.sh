#!/bin/bash -x

#- Fail early, fail often
set -e

# For debugging
printenv

# CONDA
# Install conda
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda

conda create --yes -n test python=$TRAVIS_PYTHON_VERSION pip
source activate test

# numpy scipy etc.
conda install --yes numpy scipy astropy h5py numba=0.28

# Get redrock templates
cd py/redrock
RR_TEMPLATE_VER=0.2
wget https://github.com/sbailey/redrock-templates/archive/${RR_TEMPLATE_VER}.tar.gz
tar -xzf ${RR_TEMPLATE_VER}.tar.gz
mv redrock-templates-${RR_TEMPLATE_VER} templates
rm ${RR_TEMPLATE_VER}.tar.gz
cd ../../
