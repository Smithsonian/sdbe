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
	

if __name__ == "__main__":
	# connect to redis server
	r = StrictRedis(host=REDIS_HOST,port=REDIS_PORT)
	
	# start listener
	listen_aphids = ListenerAPHIDS(r)
	listen_aphids.start()
	
	
