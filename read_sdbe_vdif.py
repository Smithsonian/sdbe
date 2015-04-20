#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  read_sdbe_vdif.py
#  Apr 01, 2015 10:33:15 HST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-04-01

"""
Define utilities for handling SWARM DBE data product.
"""

# This module is loaded from r2dbe repo on the swarm2dbe branch.
import swarm

# nump, scipy
from numpy import empty, zeros, int8, array, concatenate, ceil, arange, roll, complex64
from numpy.fft import irfft
from scipy.interpolate import interp1d

# other useful ones
from logging import getLogger
from datetime import datetime

# This module is loaded from the wideband_sw repo on the swarm_half_rate branch.
from defines import SWARM_XENG_PARALLEL_CHAN, SWARM_N_INPUTS, SWARM_N_FIDS, SWARM_TRANSPOSE_SIZE, SWARM_CHANNELS

SWARM_CHANNELS_PER_PKT = 8
SWARM_PKTS_PER_BCOUNT = SWARM_CHANNELS/SWARM_CHANNELS_PER_PKT
SWARM_SAMPLES_PER_WINDOW = 2*SWARM_CHANNELS
SWARM_RATE = 2496e6
 
# VDIF frame size
FRAME_SIZE_BYTES = 1056

#~ SWARM_XENG_PARALLEL_CHAN = 8
#~ SWARM_N_INPUTS = 2
#~ SWARM_N_FIDS = 8
#~ SWARM_TRANSPOSE_SIZE = 128
#~ SWARM_CHANNELS = 2**14
#~ SWARM_CHANNELS_PER_PKT = 8
#~ SWARM_PKTS_PER_BCOUNT = SWARM_CHANNELS/SWARM_CHANNELS_PER_PKT
#~ SWARM_SAMPLES_PER_WINDOW = 2*SWARM_CHANNELS
#~ SWARM_RATE = 2496e6

# R2DBE related constants, should probably be imported from some python
# source in the R2DBE git repo
R2DBE_SAMPLES_PER_WINDOW = 32768
R2DBE_RATE = 4096e6

def read_spectra_from_files(filename_base,bcount_offset=1,num_bcount=1):
	"""
	Read SWARM DBE spectral data from multiple files.
	
	Arguments:
	----------
	filename_base -- Base of the filenames in which the data is written.
	The collection of files take the form <filename_base>_eth<x>.vdif
	where x is one of (and all) {2,3,4,5}.
	bcount_offset -- The offset from the first encountered B-engine counter
	value, should be at least 1 (default is 1).
	num_bcount -- Read data equivalent to this many B-engine counter values
	(default is 1).
	
	Returns:
	--------
	spectra_real -- Real component of spectral data as numpy int8 array 
	and takes values in {-2,-1,0,1} or -128 for missing data. The array 
	M x 16384 where M is equal to 128*num_bcount.
	spectra_imag -- Same as spectra_real, except contains the imaginary
	component of the spectral data.
	timestamps_at_first_usable_packet -- The VDIF timestamps applied to
	the last unusable packet in each file.
	
	Notes:
	------
	Only the positive half of the spectrum is returned, as obtained from 
	a real discrete Fourier transform.
	"""

	# some benchmarking statistics
	t0_total = datetime.now()
	T_read_vdif = 0.0
	T_unpack_vdif = 0.0
	T_build_spectra = 0.0
	T_overhead = 0.0
	T_total_time = 0.0
	
	# assume all data comes from 4 seperate and equally divided load stream
	ETH_IFACE_LIST = range(2,6);
	
	# create logger for this
	logger = getLogger(__name__)
	
	
	# build filename list
	input_filenames = list()
	for i_eth_if in ETH_IFACE_LIST:
		input_filenames.append('%s_eth%d.vdif' % (filename_base,i_eth_if))
	
	logger.info('B-engine counter offset from first encountered is %d' % bcount_offset)

	num_files = len(input_filenames)
	logger.info('Found %d files for given base: %s' % (num_files,str(input_filenames)))

	spectra_real = empty([SWARM_N_INPUTS,SWARM_TRANSPOSE_SIZE*num_bcount,SWARM_CHANNELS], dtype=int8)
	spectra_imag = empty([SWARM_N_INPUTS,SWARM_TRANSPOSE_SIZE*num_bcount,SWARM_CHANNELS], dtype=int8) 
	spectra_real[:] = -128
	spectra_imag[:] = -128
	
	# set the bcount where we want to start
	bcount_start = -1
	logger.debug('bcount_start set to %d' % bcount_start)

	t0 = datetime.now()
	for this_file in input_filenames:
		with open(this_file,'r') as fh:
			this_frame_bytes = fh.read(FRAME_SIZE_BYTES)
			if (len(this_frame_bytes) < FRAME_SIZE_BYTES):
				logger.error('EoF prematurely encountered in B-engine counter scouting loop %s' % this_file)
			
			# read one frame
			this_frame = swarm.DBEFrame.from_bin(this_frame_bytes)
			# get bcount
			this_bcount = this_frame.b
			
			logger.debug('First bcount in "%s" is %d' % (this_file,this_bcount))
			
			if (this_bcount > bcount_start):
				logger.debug('This bcount is greater than global: %d > %d, updating global.' % (this_bcount,bcount_start))
				bcount_start = this_bcount
	
	# this is the B-engine count we want
	bcount_start = bcount_start + bcount_offset
	bcount_end = bcount_start + num_bcount
	T_overhead = T_overhead + (datetime.now() - t0).total_seconds()

	logger.info('Reading B-engine packets within counter values [%d,%d)' % (bcount_start,bcount_end))
	
	timestamps_at_first_usable_packet = list()
	for this_file in input_filenames:
		logger.debug('Processing file "%s"' % this_file)
		
		# reset timestamp for this stream
		timestamps_at_first_usable_packet.append(datetime(2015,01,01,0,0,0))
		with open(this_file,'r') as fh:
			while True:
				t0 = datetime.now()
				this_frame_bytes = fh.read(FRAME_SIZE_BYTES)
				T_read_vdif = T_read_vdif + (datetime.now() - t0).total_seconds()
				if (len(this_frame_bytes) < FRAME_SIZE_BYTES):
					# EoF
					logger.error('EoF prematurely encountered in data parsing loop "%s".' % this_file)
					break
				
				# build frame from bytes
				t0 = datetime.now()
				this_frame = swarm.DBEFrame.from_bin(this_frame_bytes)
				T_unpack_vdif = T_unpack_vdif + (datetime.now() - t0).total_seconds()
				
				# check the B-engine count for valid range
				if (this_frame.b < bcount_start):
					t0 = datetime.now()
					# update timestamp for this stream
					timestamps_at_first_usable_packet[-1] = this_frame.datetime()
					# skip ahead if we're from the right bcount, assuming
					# packets are in order that bcount increases monotonically
					if (this_frame.b < (bcount_start-1)):
						bcount_deficit = (bcount_start-1) - this_frame.b
						vdif_deficit = bcount_deficit*(SWARM_PKTS_PER_BCOUNT/len(ETH_IFACE_LIST))
						logger.debug('bcount is behind by at least %d whole frames, skipping %d VDIF frames' % (bcount_deficit,vdif_deficit))
						fh.seek(vdif_deficit*FRAME_SIZE_BYTES,1)
					T_overhead = T_overhead + (datetime.now() - t0).total_seconds()
					continue
				elif (this_frame.b >= bcount_end):
					# we got all the bcount
					logger.debug('bcount is %d (> %d), stop reading from %s' % (this_frame.b,bcount_end,this_file))
					break
					
				# find absolute channel positions
				t0 = datetime.now()
				chan_id = this_frame.c #chan_id
				fid = this_frame.f #fid
				start_chan = SWARM_XENG_PARALLEL_CHAN * (chan_id * SWARM_N_FIDS + fid)
				stop_chan = start_chan + SWARM_XENG_PARALLEL_CHAN
				# set time-offset of this data
				start_snap = (this_frame.b-bcount_start)*SWARM_TRANSPOSE_SIZE
				stop_snap = start_snap+SWARM_TRANSPOSE_SIZE
				
				logger.debug('bcount = %d, chan_id = %d, fid = %d: [start_snap:stop_snap, start_chan:stop_chan] = [%d:%d, %d:%d]' % (this_frame.b,chan_id,fid,start_chan,stop_chan,start_snap,stop_snap))
				
				for i_input in range(SWARM_N_INPUTS):
					p_key = 'p' + str(i_input)
					for k_parchan in range(SWARM_XENG_PARALLEL_CHAN):
						ch_key = 'ch' + str(k_parchan)
						spectra_real[i_input,start_snap:stop_snap,start_chan+k_parchan] = array(this_frame.bdata[p_key][ch_key].real,dtype=int8)
						spectra_imag[i_input,start_snap:stop_snap,start_chan+k_parchan] = array(this_frame.bdata[p_key][ch_key].imag,dtype=int8)
				
				T_build_spectra = T_build_spectra + (datetime.now() - t0).total_seconds()
	
	#~ # logging cleanup
	#~ logger.removeHandler(log_handler)
	#~ log_handler.close()
	T_total_time = T_total_time + (datetime.now() - t0_total).total_seconds()
	
	T_recording = 1.0 * num_bcount * SWARM_TRANSPOSE_SIZE * SWARM_SAMPLES_PER_WINDOW / SWARM_RATE
	
	logger.info('''Benchmark results:
	\tTotals\t\t\t\t\t\t        Time [s]\t      Per rec time
	\tUsed data recording time:\t\t\t%16.6f\t\t%10.3f
	\tTotal reading time:\t\t\t\t%16.6f\t\t%10.3f
	
	\tComponents\t\t\t\t\t        Time [s]\t      Per rec time
	\tRead raw data from file:\t\t\t%16.6f\t\t%10.3f
	\tPack into VDIF frames:\t\t\t\t%16.6f\t\t%10.3f
	\tBuild spectral data:\t\t\t\t%16.6f\t\t%10.3f
	\tOverhead:\t\t\t\t\t%16.6f\t\t%10.3f
	''' % (T_recording,T_recording/T_recording,T_total_time,T_total_time/T_recording,T_read_vdif,T_read_vdif/T_recording,T_unpack_vdif,T_unpack_vdif/T_recording,T_build_spectra,T_build_spectra/T_recording,T_overhead,T_overhead/T_recording))
	
	return spectra_real,spectra_imag,timestamps_at_first_usable_packet
	
