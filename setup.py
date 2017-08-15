#!/usr/bin/env python
# License information goes here
#
# Imports
#
import glob
import os
import sys
from setuptools import setup, find_packages
setup_keywords = dict()
#
# THESE SETTINGS NEED TO BE CHANGED FOR EVERY PRODUCT.
#
setup_keywords['name'] = 'redrock'
setup_keywords['description'] = 'Redrock redshift fitter'
setup_keywords['author'] = 'Stephen Bailey'
setup_keywords['author_email'] = 'StephenBailey@lbl.gov'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/desihub/redrock'
#
# END OF SETTINGS THAT NEED TO BE CHANGED.
#
#
# Import this module to get __doc__ and __version__.
#
sys.path.insert(int(sys.path[0] == ''),'./py')
try:
    from importlib import import_module
    product = import_module(setup_keywords['name'])
    setup_keywords['long_description'] = product.__doc__
    setup_keywords['version'] = product.__version__
except ImportError:
    #
    # Try to get the long description from the README.rst file.
    #
    if os.path.exists('README.rst'):
        with open('README.rst') as readme:
            setup_keywords['long_description'] = readme.read()
    else:
        setup_keywords['long_description'] = ''
    setup_keywords['version'] = 'unknown'
#
# Indicates if this version is a release version.
#
if setup_keywords['version'].endswith('dev'):
    #
    # Try to obtain svn information.
    #
    if 'github' not in setup_keywords['url'].lower():
        try:
            from desiUtil.install import get_svn_devstr
            setup_keywords['version'] += get_svn_devstr(setup_keywords['name'])
        except ImportError:
            pass
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
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages('py')
setup_keywords['package_dir'] = {'':'py'}
setup_keywords['test_suite'] = 'redrock.test.test_suite'

#- Load requirements.txt
# with open('requirements.txt') as fx:
#     required = list()
#     for line in fx:
#         line = line.strip()
#         if not line.startswith('#') and len(line) > 1:
#             required.append(line)
#             
# setup_keywords['install_requires'] = required

if not 'RR_TEMPLATE_DIR' in os.environ:
    setup_keywords['package_data'] = {'redrock': ['templates/*.fits']}

#
# Run setup command.
#
setup(**setup_keywords)
