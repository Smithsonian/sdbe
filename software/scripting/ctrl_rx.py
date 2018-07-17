#!/usr/bin/env python

from datetime import datetime
from os import environ
from redis import StrictRedis
from signal import SIGINT
from subprocess import Popen
from threading import Event, Thread
from time import sleep

from ctrl_shared import *

class ListenerRX(Thread):
	chan_list = [CHAN_GLOBAL, CHAN_RX]
	SGRX_CMD = "./sgrx -o {0}.vdif -a {1} -p {2} -d {3} -m {4}"
	SGRX_IP_ADDR = ["10.2.2.32","10.2.2.34"]
	SGRX_PORT = ["54323","54325"]
	SGRX_DISK_LIST = ["0,1,2,3,4,5,6,7","0,1,2,3,4,5,6,7"]
	SGRX_MOD_LIST = ["1,2","3,4"]
	
	def __init__(self, r):
		super(ListenerRX, self).__init__()
		# initialize stop event
		self._stop = Event()
		# setup redis and subscribe to channels
		self.redis = r
		self.pubsub = self.redis.pubsub()
		for chan in self.chan_list:
			self.pubsub.subscribe(chan)
		print "Channels are ", self.pubsub.channels
		# load environment for using libscatgat
		self._env = environ
		self._env['LD_LIBRARY_PATH'] = '/usr/local/lib'
		try:
			self._env['LD_LIBRARY_PATH'] = environ['LD_LIBRARY_PATH'] + ':' + self.env['LD_LIBRARY_PATH']
		except KeyError:
			None
	
	def handle_chan_rx(self,item):
		print item['channel'], "<--", item['data'], " @ ", str(datetime.now())
		if item['data'] == MSG_START:
			print "local start"
			# launch sgrx in subprocess
			self.process_start()
			# notify transmitter it can start
			sleep(5)
			r.publish(CHAN_APHIDS,MSG_START)
		elif item['data'] == MSG_STOP:
			print "local stop"
			# stop sgrx subprocess
			self.process_stop()
	
	def handle_chan_global(self,item):
		print item['channel'], "<--", item['data'], "@", str(datetime.now())
		if item['data'] == MSG_START:
			print "global start"
		elif item['data'] == MSG_STOP:
			print "global stop"
			self.stop()
	
	def process_cleanup(self):
		self._process = [None,None]
	
	def process_has_stopped(self):
		try:
			return (not self.process[0].poll() == None) and (not self.process[1].poll() == None)
		except AttributeError:
			return False
	
	def process_is_running(self):
		try:
			return (self.process[0].poll() == None) and (self.process[1].poll() == None)
		except AttributeError:
			return False
	
	def process_start(self):
		sgrx_pattern = self.redis.get(KEY_PROCESS_NAME)
		self._process = [None,None]
		for ii in xrange(2):
			print "Popen command{0}: ".format(ii), self.SGRX_CMD.format(sgrx_pattern, self.SGRX_IP_ADDR[ii], self.SGRX_PORT[ii], self.SGRX_DISK_LIST[ii], self.SGRX_MOD_LIST[ii])
			with open("stdout.log_rx{0}.{1}".format(ii,sgrx_pattern), "w") as stdout:
				with open("stderr.log_rx{0}.{1}".format(ii,sgrx_pattern), "w") as stderr:
					self._process[ii] = Popen(self.SGRX_CMD.format(sgrx_pattern, self.SGRX_IP_ADDR[ii], self.SGRX_PORT[ii], self.SGRX_DISK_LIST[ii], self.SGRX_MOD_LIST[ii]).split(), stdout=stdout, stderr=stderr, env=self.env)
	
	def process_stop(self):
		try:
			for ii in xrange(2):
				self.process[ii].send_signal(SIGINT)
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
						if item['channel'] == CHAN_RX:
							self.handle_chan_rx(item)
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
	def env(self):
		return self._env
	
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
	listen_rx = ListenerRX(r)
	listen_rx.start()
	
	
