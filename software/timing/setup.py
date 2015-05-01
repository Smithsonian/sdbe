#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  setup.py
#  Apr 29, 2015 10:58:14 EDT
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-04-29

"""
Setup script for timing module.

Compile using
	python setup.py build_ext --inplace
and import into python.
"""

from distutils.core import setup, Extension

module1 = Extension('timing',
                    sources = ['timing.c'],
                    include_dirs = ['/usr/local/include'],
                    library_dirs = ['/usr/local/lib'],
                    libraries = ['rt'])

setup (name = 'timing',
       version = '1.0',
       description = 'Provides access to UNIX timing methods.',
       ext_modules = [module1])
