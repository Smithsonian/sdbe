#!/usr/bin/env python

from datetime import datetime
from glob import glob
from os import environ
from os.path import basename, splitext
from redis import StrictRedis
from signal import SIGINT
from subprocess import Popen
from threading import Event, Lock, Thread
from time import sleep

from ctrl_shared import *

PATH_LAST_DIR="data"

class ListenerTX(Thread):
	chan_list = [CHAN_GLOBAL, CHAN_TX]
	SGTX_CMD = "./sgtx {0}.vdif {1} {2} {3}"
	SGTX_FMT_STR = "/mnt/disks/%u/%u/{0}/%s".format(PATH_LAST_DIR)
	SGTX_IP_ADDR = "192.168.10.10"
	SGTX_PORT = "12345"
	
	def __init__(self, r):
		super(ListenerTX, self).__init__()
		# initialize stop event
		self._stop = Event()
		# initialize lock object
		self._lock = Lock()
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
	
	def handle_chan_tx(self,item):
		print item['channel'], "<--", item['data'], " @ ", str(datetime.now())
		if item['data'] == MSG_START:
			print "local start"
			# launch sgtx in subprocess
			self.process_start()
		elif item['data'] == MSG_STOP:
			print "local stop"
			# stop sgtx subprocess
			self.process_stop()
	
	def handle_chan_global(self,item):
		print item['channel'], "<--", item['data'], "@", str(datetime.now())
		if item['data'] == MSG_START:
			print "global start"
		elif item['data'] == MSG_STOP:
			print "global stop"
			# stop listener thread
			listen_tx.stop()
	
	def launch_new_dataset(self,dataset):
		if self.stopped:
			return False
		# as soon as launch start, acquire the lock
		self.lock.acquire(True)
		# set the current dataset in process
		r.set(KEY_PROCESS_NAME,dataset)
		# publish receiver to start
		r.publish(CHAN_RX,MSG_START)
		return True
	
	def process_cleanup(self):
		self._process = None
		sleep(5)
		r.publish(CHAN_APHIDS,MSG_STOP)
		self.lock.release()
	
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
		sgtx_pattern = self.redis.get(KEY_PROCESS_NAME)
		print "Popen command: ", self.SGTX_CMD.format(sgtx_pattern, self.SGTX_FMT_STR, self.SGTX_IP_ADDR, self.SGTX_PORT)
		with open("stdout.log_tx.{0}".format(sgtx_pattern), "w") as stdout:
			with open("stderr.log_tx.{0}".format(sgtx_pattern), "w") as stderr:
				self._process = Popen(self.SGTX_CMD.format(sgtx_pattern, self.SGTX_FMT_STR, self.SGTX_IP_ADDR, self.SGTX_PORT).split(), stdout=stdout, stderr=stderr, env=self.env)
	
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
						if item['channel'] == CHAN_TX:
							self.handle_chan_tx(item)
						elif item['channel'] == CHAN_GLOBAL:
							self.handle_chan_global(item)
			
			# check on possibly running process
			if self.process_has_stopped():
				print "process stopped, clean up"
				self.process_cleanup()
		print "Stopping thread..."
		try:
			self.lock.release()
		except:
			print "Error: release unlocked lock"
	
	def stop(self):
		self._stop.set()
	
	def unsubscribe_all(self):
		for chan in self.chan_list:
			self.pubsub.unsubscribe(chan)
	
	def wait_until_process_done(self):
		self.lock.acquire(True)
		self.lock.release()
	
	@property
	def env(self):
		return self._env
	
	@property
	def lock(self):
		return self._lock
	
	@property
	def process(self):
		return self._process
	
	@property
	def stopped(self):
		return self._stop.isSet()

def make_dataset_list(filters):
	SEARCH_PATH="/mnt/disks/1/0/{0}/".format(PATH_LAST_DIR)
	dataset_list = []
	for f in filters:
		ls_list = glob(SEARCH_PATH+f)
		for ls_item in ls_list:
			dataset_list.append(basename(splitext(ls_item)[0]))
	return dataset_list

if __name__ == "__main__":
	from argparse import ArgumentParser
	
	parser = ArgumentParser(description="Batch process SDBE datasets with APHIDS")
	parser.add_argument('filters', nargs='*', type=str, default='*',
						help='one or more ls compatible search filters to process')
	args = parser.parse_args()
	
	# connect to redis server
	r = StrictRedis(host=REDIS_HOST,port=REDIS_PORT)
	
	# start listener
	listen_tx = ListenerTX(r)
	listen_tx.start()
	
	# get list of datasets to process -- just the VDIF filenames with
	# the extension stripped away
	dataset_list = make_dataset_list(args.filters)
	
	# publish global start
	r.publish(CHAN_GLOBAL,MSG_START)
	
	# iterate over each dataset
	for dataset in dataset_list:
		# launch new dataset
		if listen_tx.launch_new_dataset(dataset):
			print "Processing ", dataset
			# wait and toggle lock before next datasets
			listen_tx.wait_until_process_done()
		else:
			print "Unable to process ", dataset
	
	# publish global stop
	r.publish(CHAN_GLOBAL,MSG_STOP)
