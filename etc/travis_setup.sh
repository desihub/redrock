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

# CORE DEPENDENCIES
# conda install --yes pytest Cython jinja2 psutil pyyaml requests

# from the .travis.yml in https://github.com/numba
# conda install --yes -c numba llvmdev="3.7*"
# git clone git://github.com/numba/llvmlite.git -q
# cd llvmlite && python setup.py build && python setup.py install -q >/dev/null && cd ..

# numpy scipy etc.
conda install --yes numpy scipy astropy h5py numba

echo '----------------------------------------------------------------------'
echo $PYTHONPATH
which python
which nosetests
echo '----------------------------------------------------------------------'

# Get redrock templates
cd py/redrock
RR_TEMPLATE_VER=0.2
wget https://github.com/sbailey/redrock-templates/archive/${RR_TEMPLATE_VER}.tar.gz
tar -xzf ${RR_TEMPLATE_VER}.tar.gz
mv redrock-templates-${RR_TEMPLATE_VER} templates
rm ${RR_TEMPLATE_VER}.tar.gz
