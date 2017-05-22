#!/usr/bin/env python

from csv import reader
from datetime import datetime, timedelta
from logging import DEBUG, ERROR, FileHandler, Formatter, getLogger, INFO, WARNING
from numpy import angle, arange, exp, log2, nonzero, pi, polyfit, roll, sign, sqrt, unwrap
from numpy.fft import fft, irfft
from subprocess import call
from sys import exit as sys_exit
from xml.etree.ElementTree import parse

from cross_corr import corr_FXt, corr_Xt_search
from read_r2dbe_vdif import read_from_file
from vdif import UTC

# set various constants
filename_delay = "delays.csv"
path_dat = "/home/ayoung/Work/obs/Apr2017/aphids/verify/lo/dat"
path_r2dbe = "/home/ayoung/Work/obs/Apr2017/ref-ant"
path_out = "/home/ayoung/Work/obs/Apr2017/aphids/verify/lo"
path_sched = "/home/ayoung/Work/obs/Apr2017/aphids/sched"
aphids_rate = 4096e6
r2dbe_rate = 4096e6
swarm_rate = 4576e6
sdbe_spv = 32768 # samples-per-vdif
r2dbe_spv = 32768 # samples-per-vdif
sdbe_tpv = 8 # time-per-vdif, in microseconds
freq_start = 150e6 # r2dbe downconversion
tag = "12"

def get_scan_flat_files(args):
	copy_scan_cmd = "./copy_scan.sh {0.exp} {0.obs} {0.scan} {2} {0.pkt_size} {1}"
	# add some extra packets so that if we need to skip a few at the start we still have enough
	print copy_scan_cmd.format(args,path_dat,args.pkt_count+256)
	return call(copy_scan_cmd.format(args,path_dat,args.pkt_count+256).split())

def scan_to_datetime(args):
	try:
		t0 = datetime.now()
		t0 = datetime(year=t0.year,month=1,day=1,hour=0,minute=0,second=0)
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
	parser.add_argument("-s", "--ref-station-code", metavar="XX", type=str, default="Jc",
						help="two-letter station code used to record single-dish data")
	parser.add_argument("-v", "--verbose", metavar="VERBOSITY", type=int, default=0,
						help="set verbosity level to VERBOSITY, 0 = WARNING and higher, 1 = INFO and higher, 2 = DEBUG and higher (default=0)")
	parser.add_argument("exp", metavar="EXP", type=str,
						help="experiment name")
	parser.add_argument("obs", metavar="OBS", type=str,
						help="observation name")
	parser.add_argument("scan", metavar="SCAN", type=str,
						help="scan name")
	args = parser.parse_args()
	
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
	N_samples_po2 = int(log2(N_vdif_frames*r2dbe_spv))
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
	
	# first compare timestamps
	filename_aphids = "{2}/{0}_aphids-{1}.vdif".format(scan_to_name(args),tag,path_dat)
	filename_r2dbe = "r2dbe-1_if-1_{0}.vdif".format(scan_to_name(args).replace("Sm",args.ref_station_code),path_r2dbe)
	x0,v0 = read_from_file(filename_aphids,1)
	x1_full,v1 = read_from_file("{0}/{1}".format(path_r2dbe,filename_r2dbe),1)
	if v0.secs_since_epoch > v1.secs_since_epoch:
		# whoa, something wrong
		logger.info("APHIDS data v0 is late by one second or more compared to R2DBE data v1...:")
		logger.info("  v0: {0}@{1}+{2}".format(v0.ref_epoch,v0.secs_since_epoch,v0.data_frame))
		logger.info("  v1: {0}@{1}+{2}".format(v1.ref_epoch,v1.secs_since_epoch,v1.data_frame))
		sys.exit(1)
	if v0.secs_since_epoch+1 < v1.secs_since_epoch:
		# whoa, something wrong
		logger.info("APHIDS data v0 is earlier by more than one second compared to R2DBE data v1...:")
		logger.info("  v0: {0}@{1}+{2}".format(v0.ref_epoch,v0.secs_since_epoch,v0.data_frame))
		logger.info("  v1: {0}@{1}+{2}".format(v1.ref_epoch,v1.secs_since_epoch,v1.data_frame))
		sys.exit(1)
	if v0.secs_since_epoch+1 == v1.secs_since_epoch:
		# APHIDS data starts earlier, skip until next second
		N_skip = 125000 - v0.data_frame
		x0,v0 = read_from_file(filename_aphids,N_vdif_frames,offset_frames=N_skip)
		logger.info("   skipped {0} frames at start of APHIDS stream, start time is: {1}@{2}+{3}".format(N_skip,v0.ref_epoch,v0.secs_since_epoch,v0.data_frame))
		x1_full,v1 = read_from_file("{0}/{1}".format(path_r2dbe,filename_r2dbe),N_vdif_frames)
		logger.info("   skipped {0} frames at start of R2DBE stream, start time is: {1}@{2}+{3}".format(0,v1.ref_epoch,v1.secs_since_epoch,v1.data_frame))
	elif v0.secs_since_epoch == v1.secs_since_epoch:
		# R2DBE data potentially starts earlier, skip until next second
		N_skip = v0.data_frame
		x0,v0 = read_from_file(filename_aphids,N_vdif_frames)
		logger.info("   skipped {0} frames at start of APHIDS stream, start time is: {1}@{2}+{3}".format(0,v0.ref_epoch,v0.secs_since_epoch,v0.data_frame))
		x1_full,v1 = read_from_file("{0}/{1}".format(path_r2dbe,filename_r2dbe),N_vdif_frames,offset_frames=N_skip)
		logger.info("   skipped {0} frames at start of R2DBE stream, start time is: {1}@{2}+{3}".format(N_skip,v1.ref_epoch,v1.secs_since_epoch,v1.data_frame))
	
	# trim 150MHz in x1
	N_fft = 32768
	idx0_gt150MHz = int(150e6 / (4096e6/N_fft))
	X1_full = fft(x1_full.reshape((-1,N_fft)),axis=1)
	X1_trim = X1_full[:,idx0_gt150MHz:N_fft/2]
	x1 = irfft(X1_trim,axis=1,n=N_fft)
	
	# compute spectra
	x0 = x0 - x0.mean()
	X0 = fft(x0.reshape((-1,N_fft)),axis=1)
	S0 = (X0 * X0.conj()).mean(axis=0)
	x1 = x1 - x1.mean()
	X1 = fft(x1.reshape((-1,N_fft)),axis=1)
	S1 = (X1 * X1.conj()).mean(axis=0)
	f = arange(0,4096e6,4096e6/N_fft)
	
	
	#~ # remove DC
	#~ print "   removing DC in full spectra"
	#~ X_r2dbe_ip[:,[0]] = 0
	#~ X_sdbe[:,[0]] = 0
	
	# cross-correlate over wide window with lower averaging
	r = arange(-4,5)
	s_0x1,S_0x1,s_peaks = corr_Xt_search(X1[:,:16384],X0[:,:16384],fft_window_size=32768,search_range=r,search_avg=N_vdif_frames/2)
	noise = s_0x1[s_peaks.argmax(),:].std()
	signal = abs(s_0x1[s_peaks.argmax(),:]).max()
	peak_window = r[s_peaks.argmax()]
	logger.info("   cross-correlation peak of {1:.3f} with SNR of {0:.2f} in window {2}".format(signal/noise,signal,peak_window))
	try:
		assert peak_window == 0
	except AssertionError:
		logger.warning("cross-correlation peak not in zero-offset window (offset = {0})".format(peak_window))
	
	# find delay solutions
	solution_sdbe_window_lag = r[s_peaks.argmax()]
	peak_three = s_peaks[s_peaks.argmax()-1:s_peaks.argmax()+2]
	noise_three = s_0x1[s_peaks.argmax()-1:s_peaks.argmax()+2,:].std(axis=-1)
	if abs(s_0x1[s_peaks.argmax(),:]).argmax() > 16384:
		solution_sdbe_sample_lead = abs(s_0x1[s_peaks.argmax(),:]).argmax()-32768
	else:
		solution_sdbe_sample_lead = abs(s_0x1[s_peaks.argmax(),:]).argmax()
	aphids_clock_early = -solution_sdbe_window_lag*8e-6 + solution_sdbe_sample_lead/4096e6
	logger.info("   APHIDS data leads timestamp by: {0} x 8us + {1} x samples = {2:.3} ns".format(-solution_sdbe_window_lag,solution_sdbe_sample_lead,aphids_clock_early/1e-9))
	
	logger.info("   applying coarse delay solution: window lag is {0}, sample lead is {1}".format(solution_sdbe_window_lag,solution_sdbe_sample_lead))
	x1 = roll(x1,-solution_sdbe_sample_lead)
	X1 = fft(x1.reshape((-1,N_fft)),axis=1)
	S1 = (X1 * X1.conj()).mean(axis=0)
	
	r = arange(-4,5) + solution_sdbe_window_lag
	s_0x1,S_0x1,s_peaks = corr_Xt_search(X1[:,:16384],X0[:,:16384],fft_window_size=32768,search_range=r,search_avg=N_vdif_frames/2)
	noise = s_0x1[s_peaks.argmax(),:].std()
	signal = abs(s_0x1[s_peaks.argmax(),:]).max()
	peak_window = r[s_peaks.argmax()]
	logger.info("   cross-correlation peak of {1:.3f} with SNR of {0:.2f} in window {2}".format(signal/noise,signal,peak_window))
	try:
		assert signal/noise > 5
		logger.info("peak-to-std is {0} at sample {1} in window {2}".format(signal/noise,abs(s_0x1[s_peaks.argmax(),:]).argmax(),r[s_peaks.argmax()]))
	except AssertionError:
		pass_not_fail = False
		logger.error("cross-correlation peak not above 5-sigma ({0} at sample {1} in window {2})".format(signal/noise,abs(s_0x1[s_peaks.argmax(),:]).argmax(),r[s_peaks.argmax()]))
	
	logger.info("APHIDS clock is early by {0} microseconds".format(aphids_clock_early/1e-6))
	# write entry into delays file
	try:
		fh = open("{0}/{1}".format(path_out,filename_delay),"r+")
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
		fh = open("{0}/{1}".format(path_out,filename_delay),"w")
	fh.write("{0},{1},{2:10.6f}\r\n".format(meta.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),meta.end_datetime.strftime("%Y-%m-%d %H:%M:%S"),aphids_clock_early/1e-6))
	fh.close()
	
	loghndl.close()
	
	if pass_not_fail:
		sys_exit(0)
	else:
		sys_exit(2)

