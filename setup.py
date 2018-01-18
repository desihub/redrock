#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
#
# Standard imports
#
import glob
import os
import sys
#
# setuptools' sdist command ignores MANIFEST.in
#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, find_packages
#
# Begin setup
#
setup_keywords = dict()
#
# Optionally support DESI support code.
#
setup_keywords['cmdclass'] = {'sdist': DistutilsSdist}
try:
    from desiutil.setup import DesiTest, DesiVersion, get_version
    setup_keywords['cmdclass']['version'] = DesiVersion
    setup_keywords['cmdclass']['test'] = DesiTest
except ImportError:
    def get_version(productname):
        """Get the value of ``__version__`` without having to import the module.
        Parameters
        ----------
        productname : :class:`str`
            The name of the package.
        Returns
        -------
        :class:`str`
            The value of ``__version__``.
        """
        import re
        version_file = os.path.join(os.path.abspath('.'), 'py',
                                    productname, '_version.py')
        if not os.path.isfile(version_file):
            return '0.1.0'
        with open(version_file, "r") as f:
            for line in f.readlines():
                mo = re.match("__version__ = '(.*)'", line)
                if mo:
                    ver = mo.group(1)
        return ver
#
# THESE SETTINGS NEED TO BE CHANGED FOR EVERY PRODUCT.
#
setup_keywords['name'] = 'redrock'
setup_keywords['description'] = 'Redrock redshift fitter'
setup_keywords['author'] = 'DESI Collaboration'
setup_keywords['author_email'] = 'desi-data@desi.lbl.gov'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/desihub/redrock'
#
# END OF SETTINGS THAT NEED TO BE CHANGED.
#
setup_keywords['version'] = get_version(setup_keywords['name'])
#
# Use README.rst as long_description.
#
setup_keywords['long_description'] = ''
if os.path.exists('README.rst'):
    with open('README.rst') as readme:
        setup_keywords['long_description'] = readme.read()
#
# Set other keywords for the setup function.  These are automated, & should
# be left alone unless you are an expert.
#
# Treat everything in bin/ except *.rst as a script to be installed.
#
if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
        if not os.path.basename(fname).endswith('.rst')]
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>2.7.0)']
# setup_keywords['install_requires'] = ['Python (>2.7.0)']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages('py')
setup_keywords['package_dir'] = {'':'py'}
# setup_keywords['cmdclass'] = {'version': DesiVersion, 'test': DesiTest, 'sdist': DistutilsSdist}
setup_keywords['test_suite']='{name}.test.test_suite'.format(**setup_keywords)
#
# Autogenerate command-line scripts.
#
# setup_keywords['entry_points'] = {'console_scripts':['desiInstall = desiutil.install.main:main']}
#
# Add internal data directories.
#
if 'RR_TEMPLATE_DIR' not in os.environ:
    setup_keywords['package_data'] = {'redrock': ['templates/*.fits']}
#
# Run setup command.
#
setup(**setup_keywords)
