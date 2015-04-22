#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  sdbe_preprocess.py
#  Apr 09, 2015 16:29:50 EDT
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-04-09

"""
Preprocessing routines for SWARM data to match R2DBE data.
"""

import read_sdbe_vdif, read_r2dbe_vdif, cross_corr
import vdif
import numpy as np
import scipy as sp
import h5py
import logging
import struct
from string import join
import os
import sys
import resource
import matplotlib.pyplot as plt
from datetime import datetime

VDIFOUT_SAMPLES_PER_FRAME = 16384
VDIFOUT_SAMPLE_RATE = 2048e6

def run_diagnostics(scan_filename_base,rel_path_to_in='./',rel_path_to_out='./'):
	"""
	Read SBDE data, preprocess it, and write a snippet corrected data
	along with a number of diagnostics out.
	
	Arguments:
	----------
	scan_filename_base -- Base of the data filenames. SDBE data is assumed
	to be in a file named <scan_filename_base>_sdbe_eth<x>.vdif where x 
	is one of {2,3,4,5}, and R2DBE data is assumed to be in a file named 
	<scan_filename_base>_r2dbe_eth3.vdif.
	
	Returns:
	--------
	output_filename -- The name of an HDF5 file to which data was written.
	
	Notes:
	------
	Diagnostics are performed on SWARM phased sum 1 and results are assumed
	to apply directly to phased sum 0.
	
	This function does the following:
		1. Read spectrum data from SDBE data files equivalent to N_Beng_count
		B-engine counter values (128*N_Beng_count consecutive snapshots)
		2. Shift channels a-d by -2 snapshots, circularly in batches of
		128 snapshots.
		3. Read R2DBE data spanning approximately the same length of time
		as in SDBE data (N_r_vdif_frames_overbudget number of extra VDIF
		frames are included).
		4. Do an FX correlation by resampling the R2DBE data in the time-
		domain through nearest neighbour interpolation, and then subdividing
		it into SWARM-sized FFT windows. The correlation is performed 
		repeatedly, each time incrementing the relative time offset between 
		SDBE and R2DBE data by whole multiples of a SWARM FFT window to
		determine which offsets yield significant peaks. A significant
		peak is one that is peak_size_in_std_dev standard deviations above
		the mean.
		5. The peaks found in 4 are used to determine the relative time 
		offset between R2DBE and SDBE data. The time offset is divided
		into an offset in units FFT windows, and the remainder offset in 
		units samples (within one window). 
		6. The offset within one window is used to time-shift the R2DBE
		data in time so that the cross-correlation peak moves close to 
		the edge of a SWARM FFT window, either by zero-padding the R2DBE
		time-domain signal (in case SDBE data leads) or by trimming the 
		R2DBE time-domain signal (in case SDBE data lags).
		7. The FX correlation is now repeated and peaks in the cross-
		correlation determined as before. For each peak, the cross-
		correlation is calculated for a large number of snapshots using
		an averaging window of only one SWARM FFT window. This determines
		the time-reordering of SDBE snapshots.
		8. A snippet of SDBE data is time-reordered and cross-correlated
		with the time-shifted R2DBE data to demonstrate correct pre-
		processing of SDBE data. The results are logged and plots saved
		as image files.
		9. The snippet of corrected SDBE data and variables useful for
		time-reordering of SDBE data read from the raw VDIF files are
		stored in HDF5 format.
	
	Among the data stored in HDF5, the following are required to reorder
	SDBE data read from VDIF file:
		idx_first_valid -- Array of indecies indicating the first valid
		spectrum snapshots (Relative to window_search_range[idx_peaks]).
		idx_end_first_batch -- Array of indecies indicating the last
		valid spectrum snapshots in the first batch (Relative to 
		window_search_range[idx_peaks]).
		idx_start_second_batch -- Array of indecies indicating the first
		spectrum snapshots in the second batch (Relative to 
		window_search_range[idx_peaks]).
		idx_end_second_batch -- Array of indecies indicating the last 
		valid spectrum snapshots in the second batch (Relative to 
		window_search_range[idx_peaks]).
		window_search_range -- Relative offset range between R2DBE and SDBE
		data when doing cross-correlation search.
		idx_peaks -- Indecies into window_search_range that yielded 
		significant cross-correlation.
		offset_swarmdbe_data -- Number of samples (R2DBE sample rate) with
		which to offset SDBE data to time-align with R2DBE data.
	
	The snipped of SDBE data stored is formed as follows, where Xs contains
	spectrum snapshots along the 0th dimension, and spectrum channels along
	the 1st (*NOTE* I'm using pseudo-python notation here, this won't 
	work as-is):
		for ii in range(len(idx_peaks)):
			idx_first_batch = idx_first_valid[ii]:(1+idx_end_first_batch[ii])
			idx_second_batch = idx_start_second_batch[ii]:(1+idx_end_second_batch[ii])
			idx_both = concatenate(idx_first_batch,idx_second_batch)
			Xs_reordered[idx_both] = Xs[idx_both+window_search_range[idx_peaks[ii]],:]
	
	The second batch indecies tell how snapshots in one 128-snapshot range
	relate to different 128-snapshot-ranges (e.g. snapshots are in the 
	correct range, or snapshots belong to previous range and should be
	shifted by -128, etc.). This particular re-ordering is assumed to 
	be constant throughout a single record and may be used for the 
	remainder of the data. The first batch indecies include this re-
	ordering but also the (possible) truncation at the start of the
	record to align the SDBE data to R2DBE data in time. Furthermore,
	the first batch indecies do not say anything about whether and by how
	much R2DBE data may lead SDBE data. It is recommended that the first
	batch (and optionally the second batch) be read from the HDF5 file,
	and the rest from VDIF.
	
	"""
	
	# get a logger
	logger = logging.getLogger(__name__)
	
	# describe input and output paths
	logger.info('Reading from %s ... .vdif' % os.path.abspath(os.path.join(os.getcwd(), rel_path_to_in, scan_filename_base)))
	logger.info('Output to %s ... ' % os.path.abspath(os.path.join(os.getcwd(), rel_path_to_out,scan_filename_base)))
	
	# read more VDIF frames for R2DBE data than absolutely necessary in case
	# we need to offset it in time
	N_r_vdif_frames_overbudget = 10
	
	# search parameters for FX correlation
	window_search_range = np.arange(-256,256)
	window_search_avg = 128
	
	# this many standard deviations above mean defines peak in cross-correlation
	peak_size_in_std_dev = 3
	
	# realignment of SWARM data required to have correlation peak NEAR edge 
	# of R2DBE rate FFT window instead of ON edge
	offset_swarmdbe_data_overbudget = 5
	
	# Readn N_Beng_counts number of B-engine counter frames. This determines 
	# the number of R2DBE VDIF frames that need to be read to cover the same 
	# time window. The data of interest is in phased sum 1, and time realignment 
	# of B-engine packet channels is required.
	N_Beng_counts = 8
	logger.info('Reading %d B-engine counters from SWARMDBE input.' % N_Beng_counts)
	specr,speci,timestamps = read_sdbe_vdif.read_spectra_from_files(rel_path_to_in + scan_filename_base + '_swarmdbe',bcount_offset=1,num_bcount=N_Beng_counts)
	Xs1 = np.array(specr[1,:,:],dtype=np.complex64) + 1j*np.array(speci[1,:,:],dtype=np.complex64)
	Xs1 = apply_per_atoh_channel_shift(Xs1,np.array([-2,-2,-2,-2,0,0,0,0]),truncate_invalid=False,in_128=True)
	Xs0 = np.array(specr[0,:,:],dtype=np.complex64) + 1j*np.array(speci[0,:,:],dtype=np.complex64)
	Xs0 = apply_per_atoh_channel_shift(Xs0,np.array([-2,-2,-2,-2,0,0,0,0]),truncate_invalid=False,in_128=True)
	
	# check for duplicate spectrum snapshots issue
	logger.info('Produce plot of possible duplicate spectrum data.')
	idx_avg = np.array([0+i for i in range(0,1024-128,128)])
	fig, ax = plt.subplots()
	yy = abs(Xs0[idx_avg+127,:] - Xs0[idx_avg+125,:]).sum(axis=0)
	ymax = yy.max()
	ax.plot(yy,label='0-127',marker='+',color='r',linestyle='none')
	yy = abs(Xs0[idx_avg+126,:] - Xs0[idx_avg+125,:]).sum(axis=0)
	ymax = max(ymax,yy.max())
	ax.plot(yy,label='0-126',marker='o',mec='r',mfc='none',linestyle='none')
	yy = abs(Xs1[idx_avg+127,:] - Xs1[idx_avg+125,:]).sum(axis=0)
	ymax = max(ymax,yy.max())
	ax.plot(yy,label='1-127',marker='x',color='b',linestyle='none')
	yy = abs(Xs1[idx_avg+126,:] - Xs1[idx_avg+125,:]).sum(axis=0)
	ymax = max(ymax,yy.max())
	ax.plot(yy,label='1-126',marker='s',mec='b',mfc='none',linestyle='none')
	xmin = 2399
	ax.axis([xmin,xmin+26,-ymax*0.2,ymax*1.2])
	for ii in range(3):
		ax.plot(np.ones(2)*( 8-(xmin % 8) + xmin + ii*8 -0.5 ),ax.get_ylim(),'k--')
		ax.plot(np.ones(2)*( 8-(xmin % 8) + xmin + 4 + ii*8 -0.5 ),ax.get_ylim(),'k--')
		ax.text(8-(xmin % 8) + xmin + 2 + ii*8 ,-ymax*0.1,'a-d',horizontalalignment='right')
		ax.text(8-(xmin % 8) + xmin + 6 + ii*8 ,-ymax*0.1,'e-h',horizontalalignment='right')
	ax.legend(loc='upper right')
	ax.set_xlabel('Channel number')
	ax.set_ylabel('Averaged absolute difference')
	ax.set_title('For x-y: sum over B-eng counters [abs(snapshot y - snapshot 125)] for phased sum x')
	fig.canvas.draw()
	fig.savefig(rel_path_to_out + scan_filename_base + '_sdbe_preprocess_duplicate.pdf',bbox_inches='tight')
	plt.close()
	
	# Now read R2DBE data covering roughly the same time window as the SWARM
	# data. Start at an offset of zero (i.e. from the first VDIF packet) to
	# keep things simple.
	T_r_window = read_sdbe_vdif.R2DBE_RATE*read_sdbe_vdif.R2DBE_SAMPLES_PER_WINDOW
	T_s_window = read_sdbe_vdif.SWARM_RATE*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW
	N_r_vdif_frames = int(np.ceil(read_sdbe_vdif.SWARM_TRANSPOSE_SIZE*N_Beng_counts*T_r_window/T_s_window))
	N_r_vdif_frames += N_r_vdif_frames_overbudget
	vdif_frames_offset = 0
	logger.info('Reading %d VDIF frames at offset %d from R2DBE data.' % (N_r_vdif_frames,vdif_frames_offset))
	xr = read_r2dbe_vdif.read_from_file(rel_path_to_in + scan_filename_base + '_r2dbe_eth3.vdif',N_r_vdif_frames,vdif_frames_offset)

	# Time-domain interpolation and FFT to get R2DBE data in same format as
	# SWARM data.
	logger.info('Interpolating R2DBE data to SWARM rate.')
	Xr,xr_s = read_r2dbe_vdif.resample_r2dbe_to_sdbe_interp_fft(xr)

	# Do an FX correlation search to characterize the relative delay between
	# the signals and to identify required rearranging on SWARM data.
	logger.info('Measuring time misalignment via XF correlation.')
	s_0x1,S_0x1,s_peaks = cross_corr.corr_Xt_search(Xr,Xs1,search_range=window_search_range,search_avg=window_search_avg)

	# Identify peaks
	idx_peaks = np.nonzero(s_peaks - s_peaks.mean() > peak_size_in_std_dev*np.std(s_peaks))[0]

	# Time-shift the R2DBE to move peak in single window close to the edge.
	# It is assumed that the peak is at the same location in all windows.
	idx_peak_in_window = s_0x1[idx_peaks[0],:].argmax()
	
	# Record the real-time offset
	t_swarm_lead = 1.0*window_search_range[idx_peaks[0]]*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW/read_sdbe_vdif.SWARM_RATE + 1.0*idx_peak_in_window/read_sdbe_vdif.SWARM_RATE
	logger.info('Valid SWARM data leads R2DBE data by %10.6fms' % ((t_swarm_lead/1e-3)))
	
	# Apply time-shift in R2DBE data and redo cross-correlation. Pass the 
	# interpolated R2DBE signal to the method to avoid recomputing the interpolation.
	offset_r2dbe_data = idx_peak_in_window - read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW
	Xr,xr_s = read_r2dbe_vdif.resample_r2dbe_to_sdbe_interp_fft(xr,interp_kind="nearest",offset_by=offset_r2dbe_data,xr_s=xr_s)
	offset_swarmdbe_data = max(int(-offset_r2dbe_data*read_sdbe_vdif.R2DBE_RATE/read_sdbe_vdif.SWARM_RATE + offset_swarmdbe_data_overbudget),0)
	
	# Cross-correlation should now show two peaks, each at the edge of one
	# window.
	s_0x1,S_0x1,s_peaks = cross_corr.corr_Xt_search(Xr,Xs1,search_range=window_search_range,search_avg=window_search_avg)
	idx_peaks = np.nonzero(s_peaks - s_peaks.mean() > peak_size_in_std_dev*np.std(s_peaks))[0]
	#idx_peak_in_window = s_0x1[idx_peaks[0],:].argmax()
	
	logger.info('Determine correct reordering via XF correlation.')
	plot_end = 384
	idx_avg_win = np.arange(0,512)
	hi_lo_divide = np.zeros(idx_peaks.size)
	# define batches of valid indecies
	idx_first_valid = np.zeros(idx_peaks.size)
	idx_end_first_batch_valid = np.zeros(idx_peaks.size)
	idx_start_second_batch_valid = np.zeros(idx_peaks.size)
	idx_end_second_batch_valid = np.zeros(idx_peaks.size)
	idx_replace_per_peak = list()
	# initialize output
	N_valid_snapshots = Xs1.shape[0] - window_search_range[idx_peaks.max()]
	Xs1_shuffled = np.zeros((N_valid_snapshots,Xs1.shape[1]),dtype=np.complex64)
	Xs0_shuffled = np.zeros((N_valid_snapshots,Xs0.shape[1]),dtype=np.complex64)
	fig, ax = plt.subplots()
	ax.set_autoscale_on(True)
	for ii in range(len(idx_peaks)):
		S_rxs = Xr[idx_avg_win,:] * Xs1[idx_avg_win+window_search_range[idx_peaks[ii]],:].conjugate()
		s_rxs = np.fft.irfft(S_rxs,n=read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW,axis=1)
		# first do three bins, look for middle ones
		hist_,bin_edge_ = np.histogram(s_rxs[:,0],bins=3)
		if (hist_[1] == 0):
			# if no middle ones, do two bins
			hist_,bin_edge_ = np.histogram(s_rxs[:,0],bins=2)
		# either way, we're interested in the second bin edge
		hi_lo_divide[ii] = bin_edge_[1]
		# now get indecies of all points that are above the divide
		all_valids = np.nonzero(s_rxs[:,0] > hi_lo_divide[ii])[0]
		idx_diff_valid = np.diff(all_valids)
		# first set of valids may not be maximum span of valid range
		idx_first_valid[ii] = all_valids[0]
		idx_diff_greater_than_one = np.nonzero(idx_diff_valid > 1)[0][0]
		idx_end_first_batch_valid[ii] = all_valids[idx_diff_greater_than_one]
		# second set of valids will be maximum span of valid range
		idx_start_second_batch_valid[ii] = idx_end_first_batch_valid[ii] + idx_diff_valid[idx_diff_greater_than_one]
		idx_diff_greater_than_one = np.nonzero(idx_diff_valid > 1)[0][1]
		idx_end_second_batch_valid[ii] = all_valids[idx_diff_greater_than_one]
		ixd_1 = np.concatenate([np.arange(idx_start_second_batch_valid[ii],idx_end_second_batch_valid[ii]+1)+ i for i in range(0, Xs1_shuffled.shape[0], 128)])
		ixd_1 = np.concatenate([np.arange(idx_first_valid[ii],idx_end_first_batch_valid[ii]+1),ixd_1])
		ixd_1 = np.int32(ixd_1[np.nonzero(ixd_1 < N_valid_snapshots)])
		Xs1_shuffled[ixd_1,:] = Xs1[ixd_1+window_search_range[idx_peaks[ii]],:]
		Xs0_shuffled[ixd_1,:] = Xs0[ixd_1+window_search_range[idx_peaks[ii]],:]
		this_line, = ax.plot(idx_avg_win[:plot_end]+window_search_range[idx_peaks[0]],s_rxs[:plot_end,0],label="Peak at %d" % window_search_range[idx_peaks[ii]])
		ax.plot(ax.get_xlim(),np.ones(2)*hi_lo_divide[ii],'--',color=this_line.get_color())
		ax.plot(np.ones(2)*(idx_first_valid[ii]-0.25+window_search_range[idx_peaks[0]]),ax.get_ylim(),'--',color=this_line.get_color())
		ax.plot(np.ones(2)*(idx_end_first_batch_valid[ii]+0.25+window_search_range[idx_peaks[0]]),ax.get_ylim(),'--',color=this_line.get_color())
		ax.plot(np.ones(2)*(idx_start_second_batch_valid[ii]-0.25+window_search_range[idx_peaks[0]]),ax.get_ylim(),'--',color=this_line.get_color())
		ax.plot(np.ones(2)*(idx_end_second_batch_valid[ii]+0.25+window_search_range[idx_peaks[0]]),ax.get_ylim(),'--',color=this_line.get_color())
	N_valid_snapshots_limit = min((N_valid_snapshots,Xr.shape[0]))
	# Data is now shuffled into correct order, plot again correlations
	S_rxs = Xr[:plot_end+16,:] * Xs1_shuffled[:plot_end+16,:].conjugate()
	s_rxs = np.fft.irfft(S_rxs,n=read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW,axis=1)
	ax.plot(idx_avg_win[:plot_end+16]+window_search_range[idx_peaks[0]],s_rxs[:,0],'rx',label='Shuffled')
	# update plot
	ax.legend()
	ax.grid(color='gray',linestyle=':')
	ax.set_xticks(np.concatenate((idx_first_valid,np.concatenate([idx_start_second_batch_valid + i for i in range(0,plot_end,128)]))) + window_search_range[idx_peaks[0]])
	fig.canvas.draw()
	fig.savefig(rel_path_to_out + scan_filename_base + '_sdbe_preprocess_reordering.pdf',bbox_inches='tight')
	plt.close()
	
	# Convert SDBE data to time-domain to get good quantization levels
	logger.info('Transform SWARM data to time-domain to calibrate quantization levels.')
	xs0_nearest = resample_sdbe_to_r2dbe_fft_interp(Xs0_shuffled)[offset_swarmdbe_data:]
	xs1_nearest = resample_sdbe_to_r2dbe_fft_interp(Xs1_shuffled)[offset_swarmdbe_data:]
	##### This code snippet taken from alc.py
	L = len(xs0_nearest)
	y = sorted(xs0_nearest)
	# find value 16% of the way through
	Lt = int(L*0.16)
	th_1 = abs(y[Lt-1])
	# find value at 84% of the way through
	Lt2 = int(L*0.84)
	th_2 = abs(y[Lt2-1])
	# average these threshold values
	th0 = (th_1+th_2)/2
	L = len(xs1_nearest)
	y = sorted(xs1_nearest)
	# find value 16% of the way through
	Lt = int(L*0.16)
	th_1 = abs(y[Lt-1])
	# find value at 84% of the way through
	Lt2 = int(L*0.84)
	th_2 = abs(y[Lt2-1])
	# average these threshold values
	th1 = (th_1+th_2)/2
	####
	
	# Store binary data for first R2DBE VDIF packet to use as template
	# when composing SWARM data as VDIF packets
	logger.info('Storing R2DBE VDIF bytes as template.')
	f = open(rel_path_to_in + scan_filename_base + '_r2dbe_eth3.vdif','r')
	vdif_r2dbe_bytes = np.array(struct.unpack('<%dB' % read_r2dbe_vdif.FRAME_SIZE_BYTES, f.read(read_r2dbe_vdif.FRAME_SIZE_BYTES)),dtype=np.uint8)
	f.close()
	
	# print some statistics to logger
	logger.info('''First batch of valid data:
	%s
	%s
	''' % ((join(['%3d' % i for i in (idx_first_valid+window_search_range[idx_peaks])],'        ')),
	(join(['%3d' % i for i in (idx_end_first_batch_valid+window_search_range[idx_peaks])],'        '))))
	logger.info('''Second batch of valid data:
	%s
	%s
	''' % ((join(['%3d' % i for i in (idx_start_second_batch_valid+window_search_range[idx_peaks])],'        ')),
	(join(['%3d' % i for i in (idx_end_second_batch_valid+window_search_range[idx_peaks])],'        '))))
	
	# save diagnostic data to file
	output_filename = rel_path_to_out + scan_filename_base + '_sdbe_preprocess.hdf5'
	fh5 = h5py.File(output_filename,'w')
	fh5.create_dataset('Xs1',data=Xs1_shuffled)
	fh5.create_dataset('Xs0',data=Xs0_shuffled)
	fh5.create_dataset('xs1',data=xs1_nearest)
	fh5.create_dataset('xs0',data=xs0_nearest)
	fh5.create_dataset('th1',data=th1)
	fh5.create_dataset('th0',data=th0)
	fh5.create_dataset('vdif_template',data=vdif_r2dbe_bytes)
	fh5.create_dataset('offset_r2dbe_data',data=offset_r2dbe_data)
	fh5.create_dataset('offset_swarmdbe_data',data=offset_swarmdbe_data)
	fh5.create_dataset('idx_first_valid',data=idx_first_valid.astype(np.int32))
	fh5.create_dataset('idx_end_first_batch_valid',data=idx_end_first_batch_valid.astype(np.int32))
	fh5.create_dataset('idx_start_second_batch_valid',data=idx_start_second_batch_valid.astype(np.int32))
	fh5.create_dataset('idx_end_second_batch_valid',data=idx_end_second_batch_valid.astype(np.int32))
	fh5.create_dataset('idx_peaks',data=idx_peaks.astype(np.int32))
	fh5.create_dataset('window_search_range',data=window_search_range.astype(np.int32))
	fh5.close()
	
	# return the output filename
	return output_filename
	
def get_diagnostics_from_file(scan_filename_base,rel_path='./'):
	"""
	Read diagnostics from HDF5 file required to run process_chunk.
	
	Arguments:
	----------
	scan_filename_base -- Base filename for scan.
	rel_path -- Relative path to where HDF5 file is located.
	
	Returns:
	--------
	diagnostics -- Dictionary that contains the required information.
	"""
	
	diagnostics = {}
	# load diagnostic data from HDF5
	fh5 = h5py.File(rel_path + scan_filename_base + '_sdbe_preprocess.hdf5','r')
	diagnostics['offset_r2dbe_data'] = fh5.get('offset_r2dbe_data').value
	diagnostics['offset_swarmdbe_data'] = fh5.get('offset_swarmdbe_data').value
	diagnostics['start1'] = fh5.get('idx_first_valid').value.astype(int)
	diagnostics['end1'] = fh5.get('idx_end_first_batch_valid').value.astype(int)
	diagnostics['start2'] = fh5.get('idx_start_second_batch_valid').value.astype(int)
	diagnostics['end2'] = fh5.get('idx_end_second_batch_valid').value.astype(int)
	idx_peaks = fh5.get('idx_peaks').value.astype(int)
	window_search_range = fh5.get('window_search_range').value.astype(int)
	diagnostics['get_idx_offset'] = window_search_range[idx_peaks]
	diagnostics['vdif_template'] = fh5.get('vdif_template').value
	diagnostics['2bit_threshold_x0'] = fh5.get('th0').value
	diagnostics['2bit_threshold_x1'] = fh5.get('th1').value
	fh5.close()
	
	return diagnostics

def apply_per_atoh_channel_shift(Xs,shift_per_channel,truncate_invalid=False,in_128=True):
	"""
	Roll each channel along snapshots by given amounts.
	
	Arguments:
	----------
	Xs -- Spectrum snapshots as numpy array, zeroth dimension is along
	snapshots and first dimension is along frequency.
	shift_per_channel -- Array of shifts applied per channel.
	truncate_invalid -- Truncate the returned data to remove all wrapped
	content (default is False).
	in_128 -- Do a roll over each consecutive group of 128 snapshots (default
	is True). If True, then truncate_invalid is ignored.
	
	Returns:
	--------
	Xs_ret -- The spectral data after applying the necessary shifts and
	possibly truncation.
	
	Notes:
	------
	The shift per channel refers to the eight channels supplied in each
	VDIF-wrapped B-enginge packet.
	"""
	
	Xs_ret = np.zeros(Xs.shape,dtype=np.complex64)
	# apply shift per channel
	if (not in_128):
		for fid in range(read_sdbe_vdif.SWARM_N_FIDS):
			for ch_id in range(read_sdbe_vdif.SWARM_CHANNELS/(read_sdbe_vdif.SWARM_N_FIDS*read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN)):
				start_chan = ch_id*read_sdbe_vdif.SWARM_N_FIDS*read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN + fid*read_sdbe_vdif.SWARM_N_FIDS
				for ii in range(read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN):
					Xs_ret[:,start_chan+ii] = np.roll(Xs[:,start_chan+ii],shift_per_channel[ii],axis=0)
	else:
		N_passes = Xs.shape[0]/read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
		for ipass in range(N_passes):
			start_snap = ipass*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
			stop_snap = start_snap + read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
			#~ for fid in range(read_sdbe_vdif.SWARM_N_FIDS):
				#~ for ch_id in range(read_sdbe_vdif.SWARM_CHANNELS/(read_sdbe_vdif.SWARM_N_FIDS*read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN)):
					#~ start_chan = ch_id*read_sdbe_vdif.SWARM_N_FIDS*read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN + fid*read_sdbe_vdif.SWARM_N_FIDS
					#~ for ii in range(read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN):
						#~ Xs_ret[start_snap:stop_snap,start_chan+ii] = np.roll(Xs[start_snap:stop_snap,start_chan+ii],shift_per_channel[ii],axis=0)
			for ii in range(read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN):
				roll_idx = np.arange(ii,read_sdbe_vdif.SWARM_CHANNELS,read_sdbe_vdif.SWARM_XENG_PARALLEL_CHAN)
				Xs_ret[start_snap:stop_snap,roll_idx] = np.roll(Xs[start_snap:stop_snap,roll_idx],shift_per_channel[ii],axis=0)
	
	# truncate the data to remove wrapped content, ONLY if in_128 == False
	if (not in_128):
		if (truncate_invalid):
			min_shift = shift_per_channel.min()
			if (min_shift < 0):
				Xs_ret = Xs_ret[:min_shift,:]
			max_shift = shift_per_channel.max()
			if (max_shift > 0):
				Xs_ret = Xs_ret[max_shift:,:]
	
	return Xs_ret

def process_chunk(start1,end1,start2,end2,get_idx_offset,scan_filename_base,put_idx_range=[0,128]):
	"""
	Read and reorder a chunk of SBDE data.
	"""
	
	# Benchmarking
	t0_total = datetime.now()
	T_total_time = 0.0
	T_reordering = 0.0
	T_reading_beng = 0.0
	
	# make sure everythin is int
	start1 = start1.astype(int)
	start2 = start2.astype(int)
	end1 = end1.astype(int)
	end2 = end2.astype(int)
	get_idx_offset = get_idx_offset.astype(int)
	
	# initialize logger
	logger = logging.getLogger(__name__)
	
	# initialize output
	nd1 = put_idx_range[1] - put_idx_range[0]#(end1-start1+1).sum() + (num_batches-1)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
	nd2 = read_sdbe_vdif.SWARM_CHANNELS
	logger.info('Creating output array of shape %dx%d' % (nd1,nd2))
	Xs1_ordered = np.zeros((nd1,nd2),dtype=np.complex64)
	Xs0_ordered = np.zeros((nd1,nd2),dtype=np.complex64)
	
	# set some preprocessing constants
	atoh_shift_vec = np.array([-2,-2,-2,-2,0,0,0,0])
	
	# start processing on first batch
	if (put_idx_range[0] <= (end1.min()+1)):
		bcount_start = 1+ int((start1 + get_idx_offset).min()/read_sdbe_vdif.SWARM_TRANSPOSE_SIZE) # 1+ is because we always skip first bcount value
		bcount_end = 1+ int((end1 + get_idx_offset+1).max()/read_sdbe_vdif.SWARM_TRANSPOSE_SIZE)
		t0 = datetime.now()
		specr,speci,timestamps = read_sdbe_vdif.read_spectra_from_files(scan_filename_base + '_swarmdbe',bcount_offset=bcount_start,num_bcount=(bcount_end-bcount_start))
		T_reading_beng = T_reading_beng + (datetime.now() - t0).total_seconds()
		t0 = datetime.now()
		Xs1 = np.array(specr[1,:,:],dtype=np.complex64) + 1j*np.array(speci[1,:,:],dtype=np.complex64)
		Xs1 = apply_per_atoh_channel_shift(Xs1,atoh_shift_vec,truncate_invalid=False,in_128=True)
		Xs0 = np.array(specr[0,:,:],dtype=np.complex64) + 1j*np.array(speci[0,:,:],dtype=np.complex64)
		Xs0 = apply_per_atoh_channel_shift(Xs0,atoh_shift_vec,truncate_invalid=False,in_128=True)
		for ithread in range(len(start1)):
			base_idx = np.arange(start1[ithread],end1[ithread]+1)
			get_idx = base_idx+get_idx_offset[ithread]
			put_idx = base_idx
			logger.info('(1st batch) Taking data from [%d,%d] and putting it in [%d,%d]' % (get_idx[0],get_idx[-1],put_idx[0],put_idx[-1]))
			idx_valid_put_idx = np.nonzero((put_idx >= put_idx_range[0]) & (put_idx < put_idx_range[1]))[0]
			if (len(idx_valid_put_idx) == 0):
				logger.info('No valid put indecies, skipping')
				continue
			logger.info('(1st batch) Limit to data from [%d,%d] and data in [%d,%d]' % (get_idx[idx_valid_put_idx[0]],get_idx[idx_valid_put_idx[-1]],put_idx[idx_valid_put_idx[0]],put_idx[idx_valid_put_idx[-1]]))
			Xs1_ordered[put_idx[idx_valid_put_idx]-put_idx_range[0],:] = Xs1[get_idx[idx_valid_put_idx],:]
			Xs0_ordered[put_idx[idx_valid_put_idx]-put_idx_range[0],:] = Xs0[get_idx[idx_valid_put_idx],:]
		T_reordering = T_reordering + (datetime.now() - t0).total_seconds()
	
	# start processing on other batches, as for second
	base_idx_2_0 = np.arange(start2[0],end2[0]+1)
	base_idx_2_1 = np.arange(start2[1],end2[1]+1)
	batch_range = range(put_idx_range[0]/128-2,put_idx_range[1]/128+2)
	logger.info('Defined batch_range = [%d,%d)' % (batch_range[0],batch_range[-1]))
	for ibatch in batch_range:
		#~ t0 = datetime.now()
		bcount_start = 1+ int((start2 + (ibatch-1)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE + get_idx_offset).min()/read_sdbe_vdif.SWARM_TRANSPOSE_SIZE)
		bcount_end = 1+ int((end2 + (ibatch-1)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE + get_idx_offset+1).max()/read_sdbe_vdif.SWARM_TRANSPOSE_SIZE)
		#~ T_reordering = T_reordering + (datetime.now() - t0).total_seconds()
		t0 = datetime.now()
		specr,speci,timestamps = read_sdbe_vdif.read_spectra_from_files(scan_filename_base + '_swarmdbe',bcount_offset=bcount_start,num_bcount=(bcount_end-bcount_start))
		T_reading_beng = T_reading_beng + (datetime.now() - t0).total_seconds()
		t0 = datetime.now()
		Xs1 = np.array(specr[1,:,:],dtype=np.complex64) + 1j*np.array(speci[1,:,:],dtype=np.complex64)
		Xs1 = apply_per_atoh_channel_shift(Xs1,atoh_shift_vec,truncate_invalid=False,in_128=True)
		Xs0 = np.array(specr[0,:,:],dtype=np.complex64) + 1j*np.array(speci[0,:,:],dtype=np.complex64)
		Xs0 = apply_per_atoh_channel_shift(Xs0,atoh_shift_vec,truncate_invalid=False,in_128=True)
		# second batch indecies are relative to start of data, fix this since we now read from some arbitrary offset in data
		get_idx_global_offset = -int((start2 + (ibatch-1)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE + get_idx_offset).min()/read_sdbe_vdif.SWARM_TRANSPOSE_SIZE)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
		for ithread in range(len(start2)):
			if (ithread == 0):
				base_idx = base_idx_2_0 + (ibatch-1)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
			else:
				base_idx = base_idx_2_1 + (ibatch-1)*read_sdbe_vdif.SWARM_TRANSPOSE_SIZE
			get_idx = base_idx+get_idx_offset[ithread]
			get_idx = get_idx + get_idx_global_offset
			put_idx = base_idx
			logger.info('(nth batch) Taking data from [%d,%d] and putting it in [%d,%d]' % (get_idx[0],get_idx[-1],put_idx[0],put_idx[-1]))
			idx_valid_put_idx = np.nonzero((put_idx >= put_idx_range[0]) & (put_idx < put_idx_range[1]))[0]
			if (len(idx_valid_put_idx) == 0):
				logger.info('No valid put indecies, skipping')
				continue
			logger.info('(nth batch) Limit to data from [%d,%d] and data in [%d,%d]' % (get_idx[idx_valid_put_idx[0]],get_idx[idx_valid_put_idx[-1]],put_idx[idx_valid_put_idx[0]],put_idx[idx_valid_put_idx[-1]]))
			Xs1_ordered[put_idx[idx_valid_put_idx]-put_idx_range[0],:] = Xs1[get_idx[idx_valid_put_idx],:]
			Xs0_ordered[put_idx[idx_valid_put_idx]-put_idx_range[0],:] = Xs0[get_idx[idx_valid_put_idx],:]
		T_reordering = T_reordering + (datetime.now() - t0).total_seconds()
	
	T_total_time = T_total_time + (datetime.now() - t0).total_seconds()
	T_recording = 1.0 * nd1 * 32768 / read_sdbe_vdif.SWARM_RATE
	
	logger.info('''Benchmark results:
	\tTotals\t\t\t\t\t\t        Time [s]\t    Per rec time
	\tUsed data recording time:\t\t\t%16.6f\t\t%10.3f
	\tTotal time:\t\t\t\t\t%16.6f\t\t%10.3f
	
	\tComponents\t\t\t\t\t        Time [s]\t    Per rec time
	\tReading B-engine:\t\t\t\t%16.6f\t\t%10.3f
	\tRe-ordering data (PRE-preprocessing):\t\t%16.6f\t\t%10.3f
	''' % (T_recording,T_recording/T_recording,T_total_time,T_total_time/T_recording,T_reading_beng,T_reading_beng/T_recording,T_reordering,T_reordering/T_recording))
	
	return Xs1_ordered,Xs0_ordered
	
def vdif_psn_to_eud(psn):
	"""
	Return two integer values to populate eud[2:4] given the psn.
	
	Arguments:
	----------
	psn -- Packet serial number
	
	Returns:
	--------
	eud2, eud3 -- Integer values (4byte words) with which to populate
	VDIFFrame.eud[2:4].
	"""
	
	eud2 = psn & 0xffffffff
	eud3 = (psn >> 32) & 0xffffffff 
	
	return eud2,eud3
	
def vdif_station_id_str_to_int(station_id):
	"""
	Return integer value corresponding to two-letter station ID.
	
	Arguments:
	----------
	station_id -- Two letter station code.
	
	Returns:
	--------
	station_id_int -- Integer value corresponding to two-letter station code.
	"""
	
	b = struct.unpack('<2B',station_id)
	
	station_id_int = (b[0] << 8) | b[1]
	
	return station_id_int
	
def make_vdif_header_from_template(vdif_template):
	"""
	Update the VDIF header to reflect new data format.
	
	Arguments:
	----------
	template -- Numpy array of int8 that contains first R2DBE VDIF frame.
	
	Returns:
	--------
	vdf -- VDIFFrame object with all-zeros data and correct header information
	for the first frame of the processed SWARM data.
	"""
	
	# create frame from bytes
	vdf = vdif.VDIFFrame.from_bin(struct.pack('<%dB' % read_r2dbe_vdif.FRAME_SIZE_BYTES,*vdif_template))
	
	# Header information that is assumed correct are:
	#	bits-per-sample (2)
	#	complex (False)
	#	eud_vers (2)
	#	invalid_data (False)
	#	legacy_mode (False)
	#	log2_chans (0)
	#	ref_epoch (30)
	#	secs_since_epoch ...should be the same as the reference R2DBE, BUT SHOULD BE UPDATED AS FRAMES PRODUCED!
	#	thread_id (0)
	#	vdif_vers (0)
	
	# initialize data
	vdf.data = np.zeros(VDIFOUT_SAMPLES_PER_FRAME,dtype=np.int32)
	
	# frame length in 8byte chunks
	vdf.frame_length = (32 + VDIFOUT_SAMPLES_PER_FRAME*vdf.bits_per_sample/8)/8 # 32 is for VDIF header, first 8 for bits-to-bytes, second 8 because it is measured in 8byte chunks
	
	# initialize the psn to zero
	vdf.psn = 0
	# and build the eud based on that
	vdf.eud[2:4] = vdif_psn_to_eud(vdf.psn)
	
	# new sample rate
	vdf.sample_rate = VDIFOUT_SAMPLE_RATE
	
	# station ID should be 'Sm'
	vdf.station_id = vdif_station_id_str_to_int('Sm')
	
	return vdf
	
def quantize_to_2bit(xs,th):
	"""
	Quantize data to 2bit given the threshold symmetric around zero.
	
	Arguments:
	----------
	xs -- Time-domain signal to quantize.
	
	Returns:
	--------
	xs_2bit -- Time-domain signal quantized to 2bit. The returned result
	is given as a numpy array of int8.
	"""
	
	xs_out = np.zeros(xs.size,dtype=np.int8)
	idx_neg2 = np.nonzero(xs < -th)[0]
	xs_out[idx_neg2] = -2
	idx_neg1 = np.nonzero((xs >= -th) & (xs < 0))[0]
	xs_out[idx_neg1] = -1
	idx_zero = np.nonzero((xs >= 0) & (xs < th))[0]
	xs_out[idx_zero] = 0
	idx_pos1 = np.nonzero(xs >= th)[0]
	xs_out[idx_pos1] = 1
	
	return xs_out
	
def bandlimit_1248_to_1024(xs,sub_sample=True):
	"""
	Process frequency content to extract the 1024MHz from 1248MHz.
	
	Arguments:
	----------
	xs -- Time-domain signal with power-of-two number of samples greater
	than or equal to 2**12 (4096). The signal should be sampled at 4096Msps.
	sub_sample -- Flag to set whether returned result is sampled at the
	original 4096Msps or at Nyquist 2048Msps.
	
	Returns:
	--------
	xs_bl -- Signal bandlimited by extracting the desired 1024MHz.
	
	Notes:
	------
	Assumes xs is the 2nd SWARM channel so that 150MHz is discarded at
	the start, and 74MHz discarded at the end.
	"""
	
	FFT_SIZE = 2**12
	
	# lower is inclusive, upper is exclusive
	BAND_BOUNDARIES = [150e6,1174e6]
	
	# FFT and keep only positive frequencies
	Xs = np.fft.fft(xs.reshape(((xs.size/FFT_SIZE),FFT_SIZE)))[:,:FFT_SIZE/2]
	
	# band-limit
	freq_vec = np.arange(0,read_r2dbe_vdif.R2DBE_RATE/2,read_r2dbe_vdif.R2DBE_RATE/FFT_SIZE)
	idx_bl = np.nonzero((freq_vec >= BAND_BOUNDARIES[0]) & (freq_vec < BAND_BOUNDARIES[1]))[0]
	Xs_bl = Xs[:,idx_bl]
	
	# take back to time-domain
	xs_bl = np.fft.irfft(Xs_bl,n=FFT_SIZE).flatten()
	
	# optionally subsample
	if (sub_sample):
		xs_bl = xs_bl[::2]
	
	return xs_bl
	
def resample_sdbe_to_r2dbe_fft_interp(Xs,interp_kind="nearest"):
	"""
	Resample SWARM spectrum product in time-domain at R2DBE rate using
	iFFT and then interpolation in the time-domain.
	
	Arguments:
	----------
	Xs -- MxN numpy array in which the zeroth dimension is increasing
	snapshot index, and the first dimension is the positive frequency
	half of the spectrum.
	interp_kind -- Kind of interpolation. Used directly as the 'kind' 
	kwarg for scipy.interpolate.interp1d.
	
	Returns:
	--------
	xs -- The time-domain signal sampled at the R2DBE rate.
	"""
	
	# timestep sizes for SWARM and R2DBE rates
	dt_s = 1.0/read_sdbe_vdif.SWARM_RATE
	dt_r = 1.0/read_sdbe_vdif.R2DBE_RATE

	# the timespan of one SWARM FFT window
	T_s = dt_s*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW

	# the timespan of all SWARM data
	T_s_all = T_s*Xs.shape[0]

	# get time-domain signal
	xs_swarm_rate = np.fft.irfft(Xs,n=read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW,axis=1).flatten()
	# output of iFFT is periodic function, so append first sample value to the end
	xs_swarm_rate = np.concatenate((xs_swarm_rate,np.ones(1)*xs_swarm_rate[0]))
	# and calculate sample points
	t_swarm_rate = np.arange(0,xs_swarm_rate.size)*dt_s

	# calculate resample points
	t_r2dbe_rate = np.arange(0,t_swarm_rate[-1],dt_r)
	
	# and interpolate
	x_interp = sp.interpolate.interp1d(t_swarm_rate,xs_swarm_rate,kind=interp_kind)
	xs = x_interp(t_r2dbe_rate)
	
	return xs
	
def resample_sdbe_to_r2dbe_zpfft(Xs,return_xs_f=False):
	"""
	Resample SWARM spectrum product in time-domain at R2DBE rate using
	zero-padding and a radix-2 iFFT algorithm.
	
	Arguments:
	----------
	Xs -- MxN numpy array in which the zeroth dimension is increasing
	snapshot index, and the first dimension is the positive frequency
	half of the spectrum.
	return_xs_f -- Flag to indicate whether the oversampled time-domain
	signal should be returned.
	
	Returns:
	--------
	xs -- The time-domain signal sampled at the R2DBE rate.
	next_start_vec -- Start indecies for each FFT window.
	fine_sample_index -- The sample indecies used over all time to downsample
	the oversampled signal to the R2DBE rate. Only returned if 
	return_xs_f == True.
	ts_f -- Times corresponding to fine time-domain sampling. Only returned if 
	return_xs_f == True.
	xs_f -- Time-domain signal sampled at the fine rate. Only returned if 
	return_xs_f == True.
	
	"""
	
	# timestep sizes for SWARM and R2DBE rates
	dt_s = 1.0/read_sdbe_vdif.SWARM_RATE
	dt_r = 1.0/read_sdbe_vdif.R2DBE_RATE
	
	# we need to oversample by factor 64 and then undersample by factor 39
	simple_r = 64 # 4096
	simple_s = 39 # 2496
	fft_window_oversample = 2*read_sdbe_vdif.SWARM_CHANNELS*simple_r # 2* due to real FFT
	
	# oversample timestep size
	dt_f = dt_s/simple_r
	
	# the timespan of one SWARM FFT window
	T_s = dt_s*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW
	
	# what are these...?
	x_t2_0 = None
	x_t2_1 = None
	
	# time vectors over one SWARM FFT window in different step sizes
	t_r = np.arange(0,T_s,dt_r)
	t_s = np.arange(0,T_s,dt_s)
	t_f = np.arange(0,T_s,dt_f)
	
	# offset in oversampled time series that corresponds to one dt_r step
	# from the last R2DBE rate sample in the previous window
	next_start = 0
	
	# some time offsets...?
	offset_in_window_offset_s = list()
	offset_global_s = list()
	
	# total number of time series samples
	N_x = int(np.ceil(Xs.shape[0]*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW*dt_s/dt_r))
	# and initialize the output
	xs = np.zeros(N_x)
	fine_sample_index = np.zeros(N_x)
	next_start_vec = np.zeros(Xs.shape[0])
	# index in output where samples from next window are stored
	start_output = 0
	
	# store the oversampled time-domain signal if requested
	if (return_xs_f):
		ts_f = np.zeros([Xs.shape[0],read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW*simple_r])
		xs_f = np.zeros([Xs.shape[0],read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW*simple_r])
	
	for ii in range(Xs.shape[0]):
		# iFFT and oversample by 64 in one
		xs_chunk_f = np.fft.irfft(Xs[ii,:],n=fft_window_oversample)
		
		if (return_xs_f):
			ts_f[ii,:] = ii*T_s + t_f
			xs_f[ii,:] = xs_chunk_f
		
		# undersample by 39 to correct rate, and start at the correct 
		# offset in this window
		xs_chunk = xs_chunk_f[next_start::simple_s]
		stop_output = start_output+xs_chunk.size
		xs[start_output:stop_output] = xs_chunk
		fine_sample_index[start_output:stop_output] = np.arange(0,fft_window_oversample)[next_start::simple_s]
		# update the starting index in the output array
		start_output = stop_output
		
		# do we need these...?
		#~ offset_global_s.append(ii*T_s)
		#~ offset_in_window.append(next_start * dt_f)
		
		# mark the time of the last used sample relative to the start
		# of this window
		time_window_start_to_last_used_sample = t_f[next_start::39][-1]
		# calculate the remaining time in this window
		time_remaining_in_window = T_s-time_window_start_to_last_used_sample
		# convert to the equivalent number of oversample timesteps
		num_dt_f_steps_short = round(time_remaining_in_window/dt_f)
		if (num_dt_f_steps_short == 0):
			next_start = 0
		else:
			next_start = simple_s - num_dt_f_steps_short
	
	if (return_xs_f):
		return xs,next_start_vec,fine_sample_index,ts_f,xs_f
	else:
		return xs,next_start_vec
	
if __name__ == "__main__":
	
	# time
	t_total_start = datetime.now()
	
	# set the scan name
	scan_filename_base = 'prep6_test1_local'
	
	# path to diagnostics
	rel_path_out = '/home/ayoung/Work/projects/CrossCorrMar2015/out/'
	# path to VDIF data files
	rel_path_dat = '/home/ayoung/Work/projects/CrossCorrMar2015/dat/'
	
	# turn on logging
	#logging.basicConfig(format=logging.BASIC_FORMAT,level=logging.INFO,stream=sys.stdout)
	logfilename = rel_path_out + scan_filename_base + '_ppsdbe_2.log'
	logging.basicConfig(format=logging.BASIC_FORMAT,level=logging.INFO,filename=logfilename)
	logger = logging.getLogger()
	
	# read diagnostics output
	d = get_diagnostics_from_file(scan_filename_base,rel_path_out)
	
	# create a template VDIFFrame
	vdf_tmp = make_vdif_header_from_template(d['vdif_template'])
	
	# processing is done multiples of 39-snapshot frames so the time-domain 
	# signal given to bandlimit_1248_to_1024 is a power-of-two element array.
	chunks_in_39_snapshots = 32
	t_chunk = chunks_in_39_snapshots*39*read_sdbe_vdif.SWARM_SAMPLES_PER_WINDOW/read_sdbe_vdif.SWARM_RATE

	# VDIF frames per chunk -- THIS SHOULD REALLY BE A ROUND NUMBER!
	N_vdf_per_chunk = int(np.floor(t_chunk / (VDIFOUT_SAMPLES_PER_FRAME/VDIFOUT_SAMPLE_RATE)))
	frame_range = range(N_vdf_per_chunk)
	
	# initialize temporary storage for handling SWARM data offset; at 
	# most we'll want to store VDIFOUT_SAMPLES_PER_FRAME-1 samples, but
	# for now set length to zero, which indicates no buffering
	xs0_buff = np.zeros(0)
	xs1_buff = np.zeros(0)
	
	# initialize psn values
	psn_xs0 = vdf_tmp.psn
	psn_xs1 = vdf_tmp.psn
	
	# update the secs_since_epoch each time psn mods with this
	update_secs_since_epoch_per_psn = int(1.0 / (VDIFOUT_SAMPLES_PER_FRAME/VDIFOUT_SAMPLE_RATE))
	
	# real-time stop in seconds
	t_stop = 0.131072
		
	# number of chunks until we reach stop time (excluded, whole-number chunks)
	N_chunk = int(np.floor(t_stop/t_chunk))
	
	# benchmarking
	t_process = 0.0
	t_resample = 0.0
	t_bandlimit = 0.0
	t_quantize = 0.0
	t_pack_vdif = 0.0
	
	# default quantization threshold from diagnostics
	threshold_2bit_quant_xs0 = d['2bit_threshold_x0']
	threshold_2bit_quant_xs1 = d['2bit_threshold_x1']
	
	# open output file
	out_vdf_filename_xs0 = rel_path_out + scan_filename_base + '_ppsdbe_xs0_2.vdif'
	out_vdf_filename_xs1 = rel_path_out + scan_filename_base + '_ppsdbe_xs1_2.vdif'
	logger.info('Writing output VDIF to (xs0) %s and (xs1) %s' % (out_vdf_filename_xs0,out_vdf_filename_xs1))
	#~ with open(out_vdf_filename,'w') as f_vdf_out:
	try:
		f_vdf_out_xs0 = open(out_vdf_filename_xs0,'w')
		f_vdf_out_xs1 = open(out_vdf_filename_xs1,'w')
		for ichunk in range(N_chunk):
			
			# read spectrum snapshot data
			chunk_start = ichunk * chunks_in_39_snapshots * 39
			chunk_stop = (ichunk+1) * chunks_in_39_snapshots * 39
			
			logger.info('Processing chunk %d of %d: Requested index range is [%d,%d)' % (ichunk,N_chunk,chunk_start,chunk_stop))
			
			# we don't need Xs0 - only interested in one sideband for now
			t_start = datetime.now()
			Xs1,Xs0 = process_chunk(d['start1'],d['end1'],d['start2'],d['end2'],d['get_idx_offset'],rel_path_dat + scan_filename_base,put_idx_range=[chunk_start,chunk_stop])
			t_process += (datetime.now() - t_start).total_seconds()
			
			# resample in time-domain
			
			logger.info('Resampling %d SWARM FFT windows.' % Xs1.shape[0])
			
			t_start = datetime.now()
			#~ xs0 = resample_sdbe_to_r2dbe_fft_interp(Xs0,interp_kind="linear")
			xs1 = resample_sdbe_to_r2dbe_fft_interp(Xs1,interp_kind="linear")
			t_resample += (datetime.now() - t_start).total_seconds()
			
			N_pow_2 = int(np.log2(xs1.size))
			if (not (2**N_pow_2 == xs1.size)):
				logger.warn('Non-power of 2 number of resampled samples: %d != %d' % (2**N_pow_2, xs1.size))
				xs1 = xs1[:2**N_pow_2]
			
			# extract only usable part of spectrum
			
			logger.info('Bandlimiting %d time-domain samples.' % xs1.size)
			
			t_start = datetime.now()
			#~ xs0 = bandlimit_1248_to_1024(xs0,sub_sample=True)
			xs1 = bandlimit_1248_to_1024(xs1,sub_sample=True)
			t_bandlimit += (datetime.now() - t_start).total_seconds()
			
			# quantize to 2bit
			
			logger.info('Quantizing %d time-domain samples.' % xs1.size)
			
			t_start = datetime.now()
			# if this is the first chunk, determine optimal quantization thresholds
			if (ichunk == 0):
				##### This code snippet taken from alc.py
				#~ L = len(xs0)
				#~ y = sorted(xs0)
				#~ # find value 16% of the way through
				#~ Lt = int(L*0.16)
				#~ th_1 = abs(y[Lt-1])
				#~ # find value at 84% of the way through
				#~ Lt2 = int(L*0.84)
				#~ th_2 = abs(y[Lt2-1])
				#~ # average these threshold values
				#~ threshold_2bit_quant_xs0 = (th_1+th_2)/2
				L = len(xs1)
				y = sorted(xs1)
				# find value 16% of the way through
				Lt = int(L*0.16)
				th_1 = abs(y[Lt-1])
				# find value at 84% of the way through
				Lt2 = int(L*0.84)
				th_2 = abs(y[Lt2-1])
				# average these threshold values
				threshold_2bit_quant_xs1 = (th_1+th_2)/2
				#####
			
			#~ xs0 = quantize_to_2bit(xs0,threshold_2bit_quant_xs0)
			xs1 = quantize_to_2bit(xs1,threshold_2bit_quant_xs1)
			t_quantize += (datetime.now() - t_start).total_seconds()
			
			# pack in VDIF
			
			logger.info('Packing to VDIF.')
			
			t_start = datetime.now()
			# if this is the first chunk, we need to limit data taking
			# SWARM offset into account
			if (ichunk == 0):
				# divide offset by 2, since we're now at half-R2DBE-rate
				#~ xs0_swarm_offset = d['offset_swarmdbe_data']/2
				#~ xs0 = xs0[xs0_swarm_offset:]
				#~ logger.info('SWARM data offset applied to xs0: starting %d samples in, only %d samples left.' % (xs0_swarm_offset,len(xs0)))
				xs1_swarm_offset = d['offset_swarmdbe_data']/2
				xs1 = xs1[xs1_swarm_offset:]
				logger.info('SWARM data offset applied to xs1: starting %d samples in, only %d samples left.' % (xs1_swarm_offset,len(xs1)))
				
			
			#~ # append buffer if present
			#~ if (len(xs0_buff) > 0):
				#~ logger.info('Buffered data for xs0: appending %d samples to the start of %d-long xs0.' % (len(xs0_buff),len(xs0)))
				#~ xs0 = np.concatenate((xs0_buff,xs0))
			
			if (len(xs1_buff) > 0):
				logger.info('Buffered data for xs1: appending %d samples to the start of %d-long xs1.' % (len(xs1_buff),len(xs1)))
				xs1 = np.concatenate((xs1_buff,xs1))
			
			# divide into VDIF frames to write out
			#~ N_vdf_frames = int(np.ceil(1.0*max(len(xs0),len(xs1))/VDIFOUT_SAMPLES_PER_FRAME))
			N_vdf_frames = int(np.ceil(1.0*max(0,len(xs1))/VDIFOUT_SAMPLES_PER_FRAME))
			
			logger.info('Writing data to %d VDIF frames.' % N_vdf_frames)
			for iframe in range(N_vdf_frames):
				
				logger.debug('Processing VDIF frame %d of %d for chunk %d.' % (iframe,N_vdf_per_chunk,ichunk))
				
				
				
				############# NOTE THIS FLAG DEFAULT WHEN ONLY DOING ONE OUTPUT!
				full_xs0_frame = False
				#############
				
				
				
				full_xs1_frame = True
				# get range of samples to use
				samp_start = iframe*VDIFOUT_SAMPLES_PER_FRAME
				samp_stop = (iframe+1)*VDIFOUT_SAMPLES_PER_FRAME
				
				logger.debug('Samples for this frame: [%d,%d).' % (samp_start,samp_stop))
				
				#~ if (samp_stop > len(xs0)):
					#~ xs0_buff = xs0[samp_start:]
					#~ logger.info('Not enough data in xs0 to fill VDIF frame %d, buffering %d samples.' % (iframe,len(xs0_buff)))
					#~ full_xs0_frame = False
				if (samp_stop > len(xs1)):
					xs1_buff = xs1[samp_start:]
					logger.info('Not enough data in xs1 to fill VDIF frame %d, buffering %d samples.' % (iframe,len(xs1_buff)))
					full_xs1_frame = False
				
				# terminate loop if data has dried up on both streams
				if (not (full_xs0_frame or full_xs1_frame)):
					logger.info('Not enough data for full frame in either stream, breaking.')
					break
				
				if (full_xs0_frame):
					psn_xs0 += 1
					logger.debug('Writing VDIF packet %d in xs0 stream.' % psn_xs0)
					# update frame psn
					vdf_tmp.psn = psn_xs0
					# and update the eud that contains it
					vdf_tmp.eud[2:4] = vdif_psn_to_eud(vdf_tmp.psn)
					# if psn is on second boundary, then update secs_since_epoch
					if (vdf_tmp.psn % update_secs_since_epoch_per_psn == 0):
						vdf_tmp.secs_since_epoch += 1
					# update frame data for xs0 and write to file
					vdf_tmp.data = xs0[samp_start:samp_stop]
					f_vdf_out_xs0.write(vdf_tmp.to_bin())
				if (full_xs1_frame):
					psn_xs1 += 1
					logger.debug('Writing VDIF packet %d in xs1 stream.' % psn_xs1)
					# update frame psn
					vdf_tmp.psn = psn_xs1
					# and update the eud that contains it
					vdf_tmp.eud[2:4] = vdif_psn_to_eud(vdf_tmp.psn)
					# if psn is on second boundary, then update secs_since_epoch
					if (vdf_tmp.psn % update_secs_since_epoch_per_psn == 0):
						vdf_tmp.secs_since_epoch += 1
					# update frame data for xs1 and write to file
					vdf_tmp.data = xs1[samp_start:samp_stop]
					f_vdf_out_xs1.write(vdf_tmp.to_bin())
			
			t_pack_vdif += (datetime.now() - t_start).total_seconds()
			
	finally:
		
		#~ logger.info('Found %d unused samples in xs0' % len(xs0_buff))
		logger.info('Found %d unused samples in xs1' % len(xs1_buff))
		
		# close the output vdif file
		f_vdf_out_xs0.close()
		f_vdf_out_xs1.close()
	
	t_total = (datetime.now() - t_total_start).total_seconds()
	
	# output benchmarking results
	logger.info('''Benchmark results:
	\tTotals\t\t\t\t\t\t        Time [s]\t    Per rec time
	\tUsed data recording time:\t\t\t%16.6f\t\t%10.3f
	\tTotal time:\t\t\t\t\t%16.6f\t\t%10.3f

	\tComponents\t\t\t\t\t\t        Time [s]\t    Per rec time
	\tProcess:\t\t\t\t\t%16.6f\t\t%10.3f
	\tResample:\t\t\t\t\t%16.6f\t\t%10.3f
	\tBandlimit:\t\t\t\t\t%16.6f\t\t%10.3f
	\tQuantize:\t\t\t\t\t%16.6f\t\t%10.3f
	\tPack VDIF:\t\t\t\t\t%16.6f\t\t%10.3f
	''' % (t_stop,t_stop/t_stop,t_total,t_total/t_stop,t_process,t_process/t_stop,t_resample,t_resample/t_stop,t_bandlimit,t_bandlimit/t_stop,t_quantize,t_quantize/t_stop,t_pack_vdif,t_pack_vdif/t_stop))
	
	logger.info('Memory usage: %d bytes (%8.6f GB).' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2.0**20))
	
	logging.shutdown()
	
