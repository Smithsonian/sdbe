#!/usr/bin/env python

from datetime import datetime
from redis import StrictRedis
from signal import SIGINT
from subprocess import Popen, call
from threading import Event, Thread
from time import sleep

from ctrl_shared import *

class ListenerAPHIDS(Thread):
	chan_list = [CHAN_GLOBAL, CHAN_APHIDS]
	APHIDS_CMD = "aphids_start.py -i {0} --in-type {1} --inout-type {2} --out-type {3} --stdout {4} --stderr {5}"
	APHIDS_ID = "1"
	APHIDS_IN_TYPE = "net"
	APHIDS_INOUT_TYPE = "gpu"
	APHIDS_OUT_TYPE = "net"
	
	APHIDS_CMD_STOP = "aphids_stop.py {0}"
	
	def __init__(self, r):
		super(ListenerAPHIDS, self).__init__()
		# initialize stop event
		self._stop = Event()
		# setup redis and subscribe to channels
		self.redis = r
		self.pubsub = self.redis.pubsub()
		for chan in self.chan_list:
			self.pubsub.subscribe(chan)
		print "Channels are ", self.pubsub.channels
	
	def handle_chan_aphids(self,item):
		print item['channel'], "<--", item['data'], " @ ", str(datetime.now())
		if item['data'] == MSG_START:
			print "local start"
			# launch sgrx in subprocess
			self.process_start()
			# notify transmitter it can start
			sleep(15)
			r.publish(CHAN_TX,MSG_START)
		elif item['data'] == MSG_STOP:
			print "local stop"
			call(self.APHIDS_CMD_STOP.format(self.APHIDS_ID).split())
			#~ # stop sgrx subprocess
			#~ self.process_stop()
	
	def handle_chan_global(self,item):
		print item['channel'], "<--", item['data'], "@", str(datetime.now())
		if item['data'] == MSG_START:
			print "global start"
		elif item['data'] == MSG_STOP:
			print "global stop"
			self.stop()
	
	def process_cleanup(self):
		self._process = None
	
	def process_has_stopped(self):
		try:
			return not self.process.poll() == None
		except AttributeError:
			return False
	
	def process_is_running(self):
		try:
			return self.process.poll() == None
		except AttributeError:
			return False
	
	def process_start(self):
		pattern = self.redis.get(KEY_PROCESS_NAME)
		print "Popen command: ", self.APHIDS_CMD.format(self.APHIDS_ID, self.APHIDS_IN_TYPE, self.APHIDS_INOUT_TYPE, self.APHIDS_OUT_TYPE, "stdout.log_aphids.{0}".format(pattern), "stderr.log_aphids.{0}".format(pattern))
		self._process = Popen(self.APHIDS_CMD.format(self.APHIDS_ID, self.APHIDS_IN_TYPE, self.APHIDS_INOUT_TYPE, self.APHIDS_OUT_TYPE, "stdout.log_aphids.{0}".format(pattern), "stderr.log_aphids.{0}".format(pattern)).split())
	
	def process_stop(self):
		try:
			self.process.send_signal(SIGINT)
		except AttributeError:
			None
	
	def run(self):
		while not self.stopped:
			# only check for new processing if not yet done
			if not self.process_is_running():
				# check for published messages
				item = self.pubsub.get_message()
				if item:
					if item['type'] == 'message':
						if item['channel'] == CHAN_APHIDS:
							self.handle_chan_aphids(item)
						elif item['channel'] == CHAN_GLOBAL:
							self.handle_chan_global(item)
			
			# check on possibly running process
			if self.process_has_stopped():
				print "process stopped, clean up"
				self.process_cleanup()
		print "Stopping thread..."
	
	def stop(self):
		self._stop.set()
	
	def unsubscribe_all(self):
		for chan in self.chan_list:
			self.pubsub.unsubscribe(chan)
	
	@property
	def process(self):
		return self._process
	
	@property
	def stopped(self):
		return self._stop.isSet()

def map_quad_sideband_to_vdif_band(quad, sideband, frequency_band):
	sideband = sideband.upper()
	err_str = 'Unsupported frequency setup: quad={q}, sideband={s}, frequency band={f}'.format(q=quad, s=sideband, f=frequency_band)
	if frequency_band == 230:
		if sideband == "USB":
			if quad == 1:
				rx, bdc = 1, 0
				return rx, bdc
			elif quad == 2:
				rx, bdc = 1, 1
				return rx, bdc
			else:
				raise ValueError(err_str)
		elif sideband == "LSB":
			if quad == 0:
				rx, bdc = 0, 0
				return rx, bdc
			elif quad == 1:
				rx, bdc = 0, 1
				return rx, bdc
			else:
				raise ValueError(err_str)
		else:
			raise ValueError(err_str)
	elif frequency_band == 345:
		if sideband == "USB":
			if quad == 0:
				rx, bdc = 1, 0
				return rx, bdc
			elif quad == 1:
				rx, bdc = 1, 1
				return rx, bdc
			else:
				raise ValueError(err_str)
		elif sideband == "LSB":
			if quad == 0:
				rx, bdc = 0, 0
				return rx, bdc
			elif quad == 1:
				rx, bdc = 0, 1
				return rx, bdc
			else:
				raise ValueError(err_str)
		else:
			raise ValueError(err_str)
	else:
		raise ValueError(err_str)

def map_quad_sideband_to_trim(quad, sideband, frequency_band):
	sideband = sideband.upper()
	err_str = 'Unsupported frequency setup: quad={q}, sideband={s}, frequency band={f}'.format(q=quad, s=sideband, f=frequency_band)
	if frequency_band == 230:
		if sideband == "USB":
			if quad == 1:
				return 150
			elif quad == 2:
				return 150
			else:
				raise ValueError(err_str)
		elif sideband == "LSB":
			if quad == 0:
				return 102
			elif quad == 1:
				return 102
			else:
				raise ValueError(err_str)
		else:
			raise ValueError(err_str)
	elif frequency_band == 345:
		if sideband == "USB":
			if quad == 0:
				return 102
			elif quad == 1:
				return 102
			else:
				raise ValueError(err_str)
		elif sideband == "LSB":
			if quad == 0:
				return 102
			elif quad == 1:
				return 102
			else:
				raise ValueError(err_str)
		else:
			raise ValueError(err_str)
	else:
		raise ValueError(err_str)

if __name__ == "__main__":
	from argparse import ArgumentParser

	parser = ArgumentParser(description="Batch process SDBE datasets with APHIDS")
        parser.add_argument('-s', '--station-code', type=str, default='Sw',
						help='VDIF station code to use in output data (default is "Sw")')
	parser.add_argument('-q', '--quad', type=int, default=1,
						help='zero-based SWARM quadrant for this dataset (default is 1)')
	parser.add_argument('-b', '--sideband', type=str, default='USB',
						help='"USB" or "LSB" (default is "USB")')
	parser.add_argument('-f', '--frequency-band', type=int, default='230',
						help='either 230 or 345 (default is 230)')
        args = parser.parse_args()

	# connect to redis server
	r = StrictRedis(host=REDIS_HOST,port=REDIS_PORT)

	# set metadata for vdif_out_net_thread
	vdif_out_net_thread_prefix = "aphids[%d]:vdif_out_net_thread:vdif"
	rx, bdc = map_quad_sideband_to_vdif_band(args.quad, args.sideband, args.frequency_band)
	for ii in range(4):
		# station code
		key = (vdif_out_net_thread_prefix % ii) + ":station"
		value = args.station_code
		r.set(key,value)
		# BDC sideband
		key = (vdif_out_net_thread_prefix % ii) + ":bdc"
		value = str(bdc)
		r.set(key,value)
		# RX sideband
		key = (vdif_out_net_thread_prefix % ii) + ":rx"
		value = str(rx)
		r.set(key,value)

	# set the band trimming
	vdif_inout_gpu_thread_prefix = "aphids[%d]:vdif_inout_gpu_thread"
	mhz = map_quad_sideband_to_trim(args.quad, args.sideband, args.frequency_band)
	for ii in range(4):
		key = (vdif_inout_gpu_thread_prefix % ii) + ":trim_from_dc"
		value = str(mhz)
		r.set(key,value)

	# start listener
	listen_aphids = ListenerAPHIDS(r)
	listen_aphids.start()
	
	
