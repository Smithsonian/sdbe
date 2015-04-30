#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_timing.py
#  Apr 30, 2015 12:41:43 EDT
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-04-30

"""
Simple example script to illustrate the use of timing module.
"""

from numpy.random import randn

from timing import get_process_cpu_time, CLOCK_RES_PROCESS_CPU

def main():
	N_large = 100000
	N_small = 10
	
	# print clock resolution
	print "Resolution for CLOCK_PROCESS_CPUTIME_ID repored as", CLOCK_RES_PROCESS_CPU
	
	# measure large allocation
	T0 = get_process_cpu_time()
	r = randn(N_large)
	T1 = get_process_cpu_time()
	T_large = T1 - T0
	print "Time for large allocation", T_large
	
	# measure small allocation
	T0 = get_process_cpu_time()
	r = randn(N_small)
	T1 = get_process_cpu_time()
	T_small = T1 - T0
	print "Time for small allocation", T_small
	
	return 0

if __name__ == '__main__':
	main()

