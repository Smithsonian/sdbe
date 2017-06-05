#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  read_r2dbe_vdif.py
#  Apr 07, 2015 14:43:37 HST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-04-07

"""
Define utilities for handling R2DBE data product.

"""

from numpy import arange, complex64, concatenate, empty, floor, int8, log2, zeros
from logging import getLogger

import vdif

FRAME_HEADER_BYTES = 32
R2DBE_SAMPLES_PER_WINDOW = 32768
R2DBE_RATE = 4096e6

def read_from_file(filename,num_frames,offset_frames=0,samples_per_window=R2DBE_SAMPLES_PER_WINDOW):
	"""
	Read a given number of VDIF frames from file.
	
	Arguments:
	----------
	filename -- The VDIF filename from which sample data should be read.
	num_frames -- The number of frames to read.
	offset_frames -- Number of frames to skip at the start of the file.
	samples_per_window -- Samples per window (default is 32768)
	
	Returns:
	--------
	x_r2dbe -- Time-domain signal samples as a numpy array of int8 values.
	vdif0 -- VDIF header of the first used packet
	"""
	
	# get logger
	logger = getLogger(__name__)
	
	num_samples = samples_per_window * num_frames
	
	x_r2dbe = empty(num_samples,dtype=int8)
	x_r2dbe[:] = -128
	psn = 0;
	vdif0 = None
	with open(filename,'r') as f:
		# determine packet size
		b = f.read(FRAME_HEADER_BYTES)
		vh = vdif.VDIFFrameHeader.from_bin(b)
		frame_size_bytes = vh.frame_length * 8
		f.seek(0,0)
		if (offset_frames > 0):
			logger.info('Reading from offset of %d VDIF frames.' % offset_frames)
			offset_bytes = 0
			# skip all-but-one first
			f.seek(frame_size_bytes*(offset_frames-1),0)
			# read last skip and see if it is full frame
			frame_bytes = f.read(frame_size_bytes)
			if (len(frame_bytes) != frame_size_bytes):
				logger.error("EoF reached prematurely")
			logger.info('Offset by %d bytes.' % frame_size_bytes*offset_frames)
		
		for ii in range(0,num_frames):
			frame_bytes = f.read(frame_size_bytes)
			if (len(frame_bytes) != frame_size_bytes):
				logger.error("EoF reached prematurely")
				break
			frame = vdif.VDIFFrame.from_bin(frame_bytes)
			if (ii == 0):
				#~ x_r2dbe = frame.data
				vdif0 = frame
				psn = frame.psn
			else:
				#~ x_r2dbe = concatenate((x_r2dbe,frame.data))
				if not (frame.psn == (psn+1)):
					logger.warning("Packet out of order in frame %d" % ii)
				psn += 1
			x_r2dbe[ii*samples_per_window:(ii+1)*samples_per_window] = frame.data
	
	return x_r2dbe, vdif0
