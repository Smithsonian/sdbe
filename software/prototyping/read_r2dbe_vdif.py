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

# numpy and scipy
from numpy import empty, int8, floor, log2, arange, complex64, concatenate, zeros
from numpy.fft import fft
from scipy.interpolate import interp1d

# others
from logging import getLogger

# This module is loaded from the r2dbe repo on the swarm2dbe (or master?)
# branch
import vdif

# VDIF frame size
FRAME_SIZE_BYTES = 8224

# R2DBE related constants, should probably be imported from some python
# source in the R2DBE git repo
R2DBE_SAMPLES_PER_WINDOW = 32768
R2DBE_RATE = 4096e6

# SWARM related constants, should probably be imported from some python
# source in the SWARM git repo
SWARM_CHANNELS = 2**14
SWARM_SAMPLES_PER_WINDOW = 2*SWARM_CHANNELS
SWARM_RATE = 2496e6

def read_from_file(filename,num_frames,offset_frames=0,samples_per_window=R2DBE_SAMPLES_PER_WINDOW,frame_size_bytes=FRAME_SIZE_BYTES):
	"""
	Read a given number of VDIF frames from file.
	
	Arguments:
	----------
	filename -- The VDIF filename from which sample data should be read.
	num_frames -- The number of frames to read.
	offset_frames -- Number of frames to skip at the start of the file.
	samples_per_window -- Samples per window (default is 32768)
	frame_size_bytes -- Number of bytes in VDIF frame (default is 8224)
	
	Returns:
	--------
	x_r2dbe -- Time-domain signal samples as a numpy array of int8 values.
	"""
	
	# get logger
	logger = getLogger(__name__)
	
	num_samples = samples_per_window * num_frames
	
	x_r2dbe = empty(num_samples,dtype=int8)
	x_r2dbe[:] = -128
	psn = 0;
	with open(filename,'r') as f:
		if (offset_frames > 0):
			logger.info('Reading from offset of %d VDIF frames.' % offset_frames)
			offset_bytes = 0
			for ii in range(offset_frames):
				frame_bytes = f.read(frame_size_bytes)
				offset_bytes += frame_size_bytes
				if (len(frame_bytes) != frame_size_bytes):
					logger.error("EoF reached prematurely")
					break
			logger.info('Offset by %d bytes.' % offset_bytes)
		
		for ii in range(0,num_frames):
			frame_bytes = f.read(frame_size_bytes)
			if (len(frame_bytes) != frame_size_bytes):
				logger.error("EoF reached prematurely")
				break
			frame = vdif.VDIFFrame.from_bin(frame_bytes)
			if (ii == 0):
				#~ x_r2dbe = frame.data
				psn = frame.psn
			else:
				#~ x_r2dbe = concatenate((x_r2dbe,frame.data))
				if not (frame.psn == (psn+1)):
					logger.warning("Packet out of order in frame %d" % ii)
				psn += 1
			x_r2dbe[ii*samples_per_window:(ii+1)*samples_per_window] = frame.data
	
	return x_r2dbe

# this should probably be moved to somethin like r2dbe_preprocess.py or similar
def resample_r2dbe_to_sdbe_interp_fft(xr,interp_kind="nearest",offset_by=0,xr_s=None):
	"""
	Resample the R2DBE data product in the time-domain at the SWARM rate
	and FFT to the frequency domain.
	
	Arguments:
	----------
	xr -- R2DBE time-domain signal.
	interp_kind -- The interpolation method, passed directly to 
	scipy.interpolate.interp1d (default is "nearest").
	offset_by -- The number of samples to discard at the start of the
	interpolated time-domain signal prior to FX correlation. Can be negative,
	in which case the signal is zero padded by the given number of samples 
	(default is 0).
	xr_s -- Interpolated time-domain signal, if not equal to None then
	interpolation is skipped and the given array is used (default is None).
	
	Returns:
	--------
	Xr -- Two-dimensional numpy array packaged as the SWARM DBE data 
	product, except that an arbitrary number of spectral snapshots are
	available (although it will be a power of 2).
	xr_s -- R2DBE time-domain signal interpolated onto the SWARM sample
	grid assuming equal starting time.
	
	Notes:
	------
	The returned spectral snapshots only contain data for the positive
	frequency half-spectrum.
	"""
	
	if (xr_s == None):
		# define time-steps on available data
		dt_r = 1.0/R2DBE_RATE
		T_r = dt_r*xr.size
		t_r = arange(0,T_r,dt_r)
		# and get an interpolation function
		x_interp = interp1d(t_r,xr,kind=interp_kind)
		
		# define time-steps on interpolated data
		dt_s = 1.0/SWARM_RATE
		t_s = arange(0,T_r-dt_r,dt_s)
		# and interpolate
		xr_s = x_interp(t_s)
	
	# set offset in interpolated time-domain signal
	if (offset_by >= 0):
		xr_s_offset = xr_s[offset_by:]
	elif (offset_by < 0):
		xr_s_offset = concatenate((zeros(-offset_by),xr_s))
	# divide into windows and do FFT
	N_samples = 2**int(floor(log2(xr_s_offset.size)))
	Xr = complex64(fft(xr_s_offset[:N_samples].reshape((N_samples/SWARM_SAMPLES_PER_WINDOW,SWARM_SAMPLES_PER_WINDOW)),axis=1)[:,:SWARM_SAMPLES_PER_WINDOW/2])
	
	return (Xr,xr_s)
