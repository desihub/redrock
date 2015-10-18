from __future__ import absolute_import, division, print_function

import unittest

def test_suite():
    """Returns unittest.TestSuite of redrock tests for use by setup.py"""
    from os.path import dirname
    thisdir = dirname(dirname(__file__))
    return unittest.defaultTestLoader.discover(thisdir,
        top_level_dir=dirname(thisdir))
