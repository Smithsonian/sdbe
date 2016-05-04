#!/usr/bin/env python

from csv import reader
from datetime import datetime, timedelta
from logging import DEBUG, ERROR, FileHandler, Formatter, getLogger, INFO, WARNING
from numpy import arange, log2, nonzero, sign, sqrt
from numpy.fft import fft, irfft
from subprocess import call
from sys import exit as sys_exit
from xml.etree.ElementTree import parse

from cross_corr import corr_FXt
from read_r2dbe_vdif import read_from_file
from vdif import UTC

# set various constants
path_root = "/home/ayoung/Work/obs/March2016/"
filename_delay = path_root + "delays_q{0}.csv"
path_dat = path_root + "aphids-data/q{0}"
path_r2dbe = path_root + "single-dish-data/q{0}"
path_out = path_root + "verify-out/q{0}"
path_sched = path_root + "sched"
aphids_tag_list = [[],["12","34"],["34","12"]] # <-- [invalid, Quadrant1-list, Quadrant 2-list]
aphids_rate = 4096e6
r2dbe_rate = 4096e6
swarm_rate = 4160e6
sdbe_spv = 32768 # samples-per-vdif
r2dbe_spv = 32768 # samples-per-vdif
sdbe_tpv = 8 # time-per-vdif, in microseconds
freq_start = 150e6 # r2dbe downconversion
freq_stop = 2048e6 # r2dbe downconversion

def get_scan_flat_files(args):
	copy_scan_cmd = "./copy_scan.sh {0.exp} {0.obs} {0.scan} {0.swarm_quadrant} {2} {0.pkt_size} {1}".format(args,path_dat.format(args.swarm_quadrant),args.pkt_count)
	print copy_scan_cmd
	return call(copy_scan_cmd.split())

def scan_to_datetime(args):
	try:
		t0 = datetime(year=2015,month=1,day=1,hour=0,minute=0,second=0)
		day,hour_minute = args.scan.split('-')
		day = int(day)
		hour = int(hour_minute[:2])
		minute = int(hour_minute[2:])
		t1 = t0 + timedelta(day + hour/24. + minute/24./60.)
		return t1
	except:
		return None

def round_datetime_to_second(t0):
	return t0.replace(microsecond=0) + timedelta(seconds=round(t0.microsecond*1e-6))

class ScanMeta(object):
	def __init__(self,**kwargs):
		for key,val in kwargs.items():
			self.__setattr__(key,val)
	
	def datetime_to_vextime(self,dt):
		vt = datetime.strftime(dt,"%Yy%jd%Hh%Mm%Ss")
		return vt
	
	@property
	def end_datetime(self):
		return self.start_datetime + timedelta(int(self.duration) * 1/24./60./60.)
	
	@property
	def end_vextime(self):
		return self.datetime_to_vextime(self.end_datetime)

	@property
	def start_datetime(self):
		return datetime.strptime(self.start_time,"%Y%j%H%M%S").replace(tzinfo=UTC())
	
	@property
	def start_vextime(self):
		return self.datetime_to_vextime(self.start_datetime)

def get_scan_meta_info(args):
	global logger
	try:
		
		tree = parse(path_sched + "/" + args.exp + ".xml")
		root = tree.getroot()
		element_scan = root.find("./scan[@scan_name='{0}']".format(args.scan))
	except IOError:
		logger.warning("No schedule XML found for exp '{0}'".format(args.exp))
		return None
	meta = ScanMeta(**element_scan.attrib)
	return meta

def scan_to_name(args):
	return "{0.exp}_{0.obs}_{0.scan}".format(args)

def scan_to_single_dish_filename(args):
	return "{0.exp}_{0.obs}_{0.scan}_r2dbe_q{0.swarm_quadrant}.vdif".format(args); 

def scan_to_aphids_filename(args):
	return "{0.exp}_{0.obs}_{0.scan}_aphids-{1}_q{0.swarm_quadrant}.vdif".format(args,"{0}"); 

if __name__ == "__main__":
	from argparse import ArgumentParser
	
	# parse input arguments
	parser = ArgumentParser(description=
		"Verify APHIDS preprocessed SMA phased array scan by cross-correlation with SMA single dish data",
							epilog=
		"This script uses copy_scan.sh (locally) and setup_scan.sh (remotely), see documentation for these scripts for more information. "
		"Scan flat-file name are composed as ${EXP}_${OBS}_${SCAN} with a suffix '_aphids-12' or '_aphids-34' where appropriate, and an extension '.vdif'."
	)
	parser.add_argument("-b", "--pkt-size", metavar="PKTSIZE", type=int, default=8224,
						help="size of VDIF packet in bytes (default=4128)")
	parser.add_argument("-c", "--pkt-count", metavar="PKTCOUNT", type=int, default=1024,
						help="number of VDIF packets to slice in each dataset (default=1024)")
	parser.add_argument("-v", "--verbose", metavar="VERBOSITY", type=int, default=0,
						help="set verbosity level to VERBOSITY, 0 = WARNING and higher, 1 = INFO and higher, 2 = DEBUG and higher (default=0)")
	parser.add_argument("-q", "--swarm-quadrant", metavar="QUAD", type=int, default=0,
						help="dataset from SWARM quadrant QUAD, should be 0, 1 or 2 (default=0, means don't group into quadrant)")
	parser.add_argument("exp", metavar="EXP", type=str,
						help="experiment name")
	parser.add_argument("obs", metavar="OBS", type=str,
						help="observation name")
	parser.add_argument("scan", metavar="SCAN", type=str,
						help="scan name")
	args = parser.parse_args()
	
	if args.swarm_quadrant == 0:
		raise ValueError("This script now requires the SWARM quadrant to be specified, either '1' or '2'.")
	
	# set all quadrant-dependent values
	path_dat = path_dat.format(args.swarm_quadrant)
	path_r2dbe = path_r2dbe.format(args.swarm_quadrant)
	filename_r2dbe = path_r2dbe + "/" + scan_to_single_dish_filename(args)
	path_out = path_out.format(args.swarm_quadrant)
	filename_delay = filename_delay.format(args.swarm_quadrant)
	aphids_tag_list = aphids_tag_list[args.swarm_quadrant]
	filename_aphids = path_dat + "/" + scan_to_aphids_filename(args)
	
	# global logger
	logger = getLogger(__name__)
	formatter = Formatter("%(levelname)s - %(asctime)s - %(message)s")
	loghndl = FileHandler("{1}/{0}.log".format(scan_to_name(args),path_out),mode="w")
	loghndl.setFormatter(formatter)
	logger.addHandler(loghndl)
	if args.verbose <= 0:
		logger.setLevel(WARNING)
	elif args.verbose == 1:
		logger.setLevel(INFO)
	elif args.verbose >= 2:
		logger.setLevel(DEBUG)
	logger.info("start logging")
	
	# set all constants
	N_vdif_frames = args.pkt_count
	sdbe_bpv = args.pkt_size
	logger.info("reading {0} VDIF frames".format(N_vdif_frames))
	
	# copy processed scan flat files
	if get_scan_flat_files(args):
		logger.error("could not retrieve flat file for '{0.exp} {0.obs} {0.scan}'".format(args))
		sys_exit(1)
	
	# get datetime for scan start
	time_start = scan_to_datetime(args)
	
	# get scan meta information
	meta = get_scan_meta_info(args)
	
	# initialize return code
	pass_not_fail = True
	
	# read APHIDS data
	xa,vdifa = [None,None],[None,None]
	tag_list = aphids_tag_list
	for tag in tag_list:
		idx = tag_list.index(tag)
		this_filename = filename_aphids.format(tag)
		logger.debug("read APHIDS data from {0}".format(this_filename))
		xa[idx],vdifa[idx] = read_from_file(this_filename,N_vdif_frames,0,
									samples_per_window=sdbe_spv,frame_size_bytes=sdbe_bpv)
		# fix sample rate
		vdifa[idx].sample_rate = aphids_rate
		logger.info("APHIDS VDIF timestamp chan{0} is {1}".format(idx,str(vdifa[idx].datetime())))
	if meta:
		timedelta_offset = vdifa[0].datetime() - meta.start_datetime
	else:
		timedelta_offset = vdifa[0].datetime() - round_datetime_to_second(vdifa[0].datetime())
	offset_sign = 1 if timedelta_offset.days == 0 else sign(timedelta_offset.days)
	N_vdif_offset_r2dbe = offset_sign*abs(timedelta_offset).microseconds/sdbe_tpv
	logger.debug("scan metadata {1}available, N_vdif_offset_r2dbe = {0}".format(N_vdif_offset_r2dbe,"" if meta else "un"))
	if N_vdif_offset_r2dbe < 0:
		N_vdif_offset_aphids = -N_vdif_offset_r2dbe
		logger.info("re-read APHIDS data with offset of {0} (negative R2DBE offset)".format(N_vdif_offset_aphids))
		# re-read APHIDS data
		for tag in tag_list:
			idx = tag_list.index(tag)
			this_filename = filename_aphids.format(tag)
			logger.debug("read APHIDS data from {0}".format(this_filename))
			xa[idx],vdifa[idx] = read_from_file(this_filename,N_vdif_frames,N_vdif_offset_aphids,
										samples_per_window=sdbe_spv,frame_size_bytes=sdbe_bpv)
			# fix sample rate
			vdifa[idx].sample_rate = aphids_rate
			logger.info("APHIDS VDIF timestamp chan{0} is {1}".format(idx,str(vdifa[idx].datetime())))
		N_vdif_offset_r2dbe = 0
		logger.debug("negative offset, correct N_vdif_offset_r2dbe <-- {0}".format(N_vdif_offset_r2dbe))
	
	# read single dish data
	logger.debug("read R2DBE data from {0}".format(filename_r2dbe))
	xr,vdifr = read_from_file(filename_r2dbe,N_vdif_frames,N_vdif_offset_r2dbe)
	logger.info("R2DBE VDIF timestamp is {0}".format(str(vdifr.datetime())))
	
	# do digital downconversion of single dish data
	Xr = fft(xr.reshape((N_vdif_frames,r2dbe_spv)))[:,:r2dbe_spv/2]
	fr = arange(0,r2dbe_rate/2,r2dbe_rate/r2dbe_spv)
	idx_start = nonzero(fr >= freq_start)[0][0]
	Xr_sub = Xr[:,idx_start:]
	xr = irfft(Xr_sub,n=sdbe_spv,axis=1).flatten()
	
	# cross-correlate
	search_range = arange(-8,8)
	search_avg = N_vdif_frames-2*abs(search_range).max()
	logger.debug("search range [{0},{1}] and averaging over {2} windows".format(search_range.min(),search_range.max(),search_avg))
	s_a0xr,S_a0xr,p_a0xr = corr_FXt(xa[0],xr,fft_window_size=sdbe_spv,search_avg=search_avg,search_range=search_range)
	s_a1xr,S_a1xr,p_a1xr = corr_FXt(xa[1],xr,fft_window_size=sdbe_spv,search_avg=search_avg,search_range=search_range)
	
	# check peak is in zero-window offset
	try:
		assert search_range[p_a1xr.argmax()] == 0
	except AssertionError:
		logger.warning("cross-correlation peak not in zero-offset window (offset = {0})".format(search_range[p_a1xr.argmax()]))
	
	# check peak is above threshold
	try:
		peak = abs(s_a1xr[p_a1xr.argmax(),:]).max()
		sigma = s_a1xr[p_a1xr.argmax(),:].std()
		assert peak/sigma > 5 # looking for at least >5sigma peak, otherwise something very wrong
		logger.info("peak-to-std is {0} at sample {1} in window {2}".format(peak/sigma,abs(s_a1xr[p_a1xr.argmax(),:]).argmax(),search_range[p_a1xr.argmax()]))
	except AssertionError:
		pass_not_fail = False
		logger.error("cross-correlation peak not above 5-sigma ({0} at sample {1} in window {2})".format(peak/sigma,abs(s_a1xr[p_a1xr.argmax(),:]).argmax(),search_range[p_a1xr.argmax()]))
	
	# try to correct delay
	sample_offset = abs(s_a1xr[p_a1xr.argmax(),:]).argmax()
	# first delay aphids data
	s_tmp,S_tmp,p_tmp = corr_FXt(xa[1][sample_offset:],xr,fft_window_size=sdbe_spv,search_avg=search_avg-1,search_range=search_range)
	logger.debug("assuming APHIDS data leads: peak-to-std is {0} at offset {1} in window {2} (after delay correction)".format(peak/sigma,abs(s_tmp[p_tmp.argmax(),:]).argmax(),search_range[p_tmp.argmax()]))
	if abs(s_tmp[p_tmp.argmax(),:]).argmax() != 0:
		sample_offset = sample_offset-sdbe_spv
		s_tmp,S_tmp,p_tmp = corr_FXt(xa[1],xr[-sample_offset:],fft_window_size=sdbe_spv,search_avg=search_avg-1,search_range=search_range)
		logger.debug("assuming R2DBE data leads: peak-to-std is {0} at offset {1} in window {2} (after delay correction)".format(peak/sigma,abs(s_tmp[p_tmp.argmax(),:]).argmax(),search_range[p_tmp.argmax()]))
	
	# correct delay if we're outside 0 offset window
	if search_range[p_tmp.argmax()] != 0:
		# APHIDS clock offset possibly larger than single window?
		sample_offset_whole_windows = -search_range[p_tmp.argmax()]*sdbe_spv
		logger.debug("delay is multiple windows, applying whole-window sample offset of {0}".format(sample_offset_whole_windows))
		sample_offset += sample_offset_whole_windows
		if sample_offset > 0:
			s_tmp,S_tmp,p_tmp = corr_FXt(xa[1][sample_offset:],xr,fft_window_size=sdbe_spv,search_avg=search_avg-1-abs(search_range[p_a1xr.argmax()]),search_range=search_range)
		else:
			s_tmp,S_tmp,p_tmp = corr_FXt(xa[1],xr[-sample_offset:],fft_window_size=sdbe_spv,search_avg=search_avg-1-abs(search_range[p_a1xr.argmax()]),search_range=search_range)
	
	aphids_clock_early = sample_offset / aphids_rate
	logger.info("APHIDS clock is early by {0} microseconds".format(aphids_clock_early/1e-6))
	
	# write entry into delays file
	try:
		fh = open(filename_delay,"r+")
		while True:
			pos = fh.tell()
			line = fh.readline()
			if not line:
				break
			csv_reader = reader([line])
			timestamp = datetime.strptime(csv_reader.next()[0],"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC())
			if int((timestamp - meta.start_datetime).total_seconds()) == 0:
				fh.seek(pos,0)
				break
	except IOError:
		fh = open(filename_delay,"w")
	fh.write("{0},{1},{2:10.6f}\r\n".format(meta.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),meta.end_datetime.strftime("%Y-%m-%d %H:%M:%S"),aphids_clock_early/1e-6))
	fh.close()
	
	# check peak is in zero-window offset after delay correction
	try:
		assert search_range[p_tmp.argmax()] == 0
	except AssertionError:
		pass_not_fail = False
		logger.error("cross-correlation peak not in zero-offset window (after delay correction, offset = {0})".format(search_range[p_a1xr.argmax()]))
	
	# check peak is at sample zero
	try:
		assert abs(s_tmp[p_tmp.argmax(),:]).argmax() == 0
	except AssertionError:
		pass_not_fail = False
		logger.error("cross-correlation peak not at sample zero (after delay correction, sample = {0})".format(s_tmp[p_tmp.argmax(),:].argmax()))
	
	# check peak-to-noise is sensible
	try:
		peak = abs(s_tmp[p_tmp.argmax(),:]).max()
		sigma = s_tmp[p_tmp.argmax(),:].std()
		assert peak/sigma > 5 # looking for at least >5sigma peak, otherwise something very wrong
		logger.info("peak-to-std is {0} at sample {1} in window {2} (after delay correction)".format(peak/sigma,abs(s_tmp[p_tmp.argmax(),:]).argmax(),search_range[p_tmp.argmax()]))
	except AssertionError:
		pass_not_fail = False
		logger.error("cross-correlation peak not above 5-sigma (after delay correction, peak-to-std = {0})".format(peak/sigma))
	
	loghndl.close()
	
	if pass_not_fail:
		sys_exit(0)
	else:
		sys_exit(2)
#~ 
#~ Ss0 = cross_corr.corr_Xt(Xs0,Xs0,fft_window_size=sdbe_spv)[1]
#~ Ss1 = cross_corr.corr_Xt(Xs1,Xs1,fft_window_size=sdbe_spv)[1]
#~ Sr = cross_corr.corr_FXt(xr,xr,fft_window_size=sdbe_spv)[1]
#~ Sa0 = cross_corr.corr_FXt(xa0,xa0,fft_window_size=sdbe_spv)[1]
#~ Sa1 = cross_corr.corr_FXt(xa1,xa1,fft_window_size=sdbe_spv)[1]
#~ 
#~ freq_r2dbe = 8150e6 - np.arange(0,read_r2dbe_vdif.R2DBE_RATE/2,read_r2dbe_vdif.R2DBE_RATE/2/(sdbe_spv/2))
#~ freq_sdbe = np.arange(0,read_sdbe_vdif.SWARM_RATE/2,read_sdbe_vdif.SWARM_RATE/2/read_sdbe_vdif.SWARM_CHANNELS)
#~ freq_sdbe_s0 = freq_sdbe + 7850e6
#~ freq_sdbe_s1 = 8150e6 - freq_sdbe
#~ freq_aphids = np.arange(0,read_r2dbe_vdif.R2DBE_RATE/4,read_r2dbe_vdif.R2DBE_RATE/4/(sdbe_spv/2))
#~ freq_aphids_a0 = freq_aphids + 8000e6
#~ freq_aphids_a1 = 8000e6 - freq_aphids
#~ 
#~ # plot power spectral densities
#~ fig = plt.figure()
#~ ax = plt.subplot(111)
#~ ax.plot(freq_sdbe_s0/1e6,abs(Ss0)/abs(Ss0[1:]).mean(),'b-',label='SDBE ch0 (1st IF)')
#~ ax.plot(freq_sdbe_s1/1e6,abs(Ss1)/abs(Ss1[1:]).mean(),'r-',label='SDBE ch1 (1st IF)')
#~ ax.plot(freq_aphids_a0/1e6,abs(Sa0)/abs(Sa0[1:]).mean()/2,'b:',label='APHIDS ch0 (1st IF)')
#~ ax.plot(freq_aphids_a1/1e6,abs(Sa1)/abs(Sa1[1:]).mean()/2,'r:',label='APHIDS ch1 (1st IF)')
#~ ax.plot(freq_r2dbe/1e6,abs(Sr)/abs(Sr).mean()/3*2,'g:',label='Single dish ref (1st IF)')
#~ ax.set_xlabel('Frequency [MHz]')
#~ ax.set_ylabel('Power spectral density [arbitrary units]')
#~ ax.set_title('Bandpass of SWARM data in first IF before and after preprocessing with APHIDS')
#~ ax.set_ylim([0,2])
#~ ax.set_xlim([6000,9500])
#~ ax.legend()
#~ 
#~ # plot cross-correlation phase
#~ s_a1xr,S_a1xr,p_a1xr = cross_corr.corr_FXt(xa1[8721:],xr,fft_window_size=sdbe_spv,search_avg=128)
#~ fig = plt.figure()
#~ ax = plt.subplot(211)
#~ ax.plot(freq_aphids_a1/1e6,abs(S_a1xr)/abs(S_a1xr).mean(),'b-',label='amplitude')
#~ ax.set_xlabel('Frequency [MHz]')
#~ ax.set_ylabel('Cross-power spectral density [arbitrary units]')
#~ ax.set_title('Cross-correlation in spectral domain amplitude and phase')
#~ ax.set_ylim([0,4])
#~ ax.set_xlim([6750,8250])
#~ ax.legend()
#~ fig = plt.subplot(212)
#~ ax = fig.axes
#~ ax.plot(freq_aphids_a1/1e6,np.angle(S_a1xr),'bx',label='phase')
#~ ax.set_xlabel('Frequency [MHz]')
#~ ax.set_ylabel('Cross-power spectral density [radians]')
#~ ax.set_ylim([-3.5,3.5])
#~ ax.set_xlim([6750,8250])
#~ ax.legend()
