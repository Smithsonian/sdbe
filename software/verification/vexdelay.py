#!/usr/bin/env python

from csv import reader
from datetime import datetime, timedelta, tzinfo
from sys import version_info

vex_time_fmt="%Yy%jd%Hh%Mm%Ss"

class CustomTimeDelta(timedelta):
	def custom_total_seconds(self):
		if version_info >= (2,7):
			#~ print ">= 2.7, using built-in total_seconds"
			return self.total_seconds()
		else:
			#~ print "< 2.7, using local total_seconds"
			return self.days*24*60*60 + self.seconds + self.microseconds*1e-6

class UTC(tzinfo):
	""" UTC tzinfo """
	
	def utcoffset(self, dt):
		return timedelta(0)
	
	def tzname(self, dt):
		return "UTC"
	
	def dst(self, dt):
		return timedelta(0)

class ClockDef():
	def __init__(self,start,stop,early,rate):
		self._start = start
		self._stop = stop
		self._early = early
		self._rate = rate
	
	@classmethod
	def from_csv_file(cls,start,stop,filename="delays.csv",offset=0.0,rate=0.0):
		clks = []
		with open(filename,"r") as fh:
			lines = fh.readlines()
			csv_reader = reader(lines)
			for row in csv_reader:
				r_start = datetime.strptime(row[0],"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC())
				r_stop = datetime.strptime(row[1],"%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC())
				#~ if (start - r_stop).total_seconds() < 0.0 and (stop - r_start).total_seconds() > 0.0:
					#~ clks.append(cls(r_start,r_stop,float(row[2])+offset,rate))
				start_min_r_stop = start - r_stop;
				start_min_r_stop_ctd = CustomTimeDelta(start_min_r_stop.days,start_min_r_stop.seconds,start_min_r_stop.microseconds);
				stop_min_r_start = stop - r_start;
				stop_min_r_start_ctd = CustomTimeDelta(stop_min_r_start.days,stop_min_r_start.seconds,stop_min_r_start.microseconds);
				if start_min_r_stop_ctd.custom_total_seconds() < 0.0 and stop_min_r_start_ctd.custom_total_seconds() > 0.0:
					clks.append(cls(r_start,r_stop,float(row[2])+offset,rate))
		return sorted(clks,cmp=cls.cmp_clk_def)
	
	@classmethod
	def cmp_clk_def(cls,clk1,clk2):
		if clk1.start > clk2.start:
			return 1
		else:
			if clk1.start == clk2.start:
				return 0
			else:
				return -1
	
	def datetime_to_vextime(self,dt):
		vt = dt.strftime(vex_time_fmt)
		return vt
	
	def toVex(self):
		return "clock_early={0} : {1:7.3f} usec : {2} : {3} ;\r\n".format(
					self.datetime_to_vextime(self.start),
					self.early,
					self.datetime_to_vextime(self.start),#<<--- second timestamp is epoch (zero-point solution), which in this case is exactly the same as the solution starting time. self.datetime_to_vextime(self.stop),
					self.rate)
	
	def add_delay(self,delay):
		self._early = self.early - delay
	
	@property
	def early(self):
		return self._early
	
	@property
	def rate(self):
		return self._rate
	
	@property
	def start(self):
		return self._start
	
	@property
	def stop(self):
		return self._stop
	

def vex_block(clockdefs,station_id="Sm"):
	str = "def {0};\r\n".format(station_id)
	for clockdef in clockdefs:
		str += "  " + clockdef.toVex()
	str += "enddef;\r\n"
	return str

def run_programmatic_example():
	# test a few scans from day 89, from 08h00-09h00 
	start = datetime(2015,3,30,8,00,0,tzinfo=UTC())
	stop =  datetime(2015,3,30,9,00,0,tzinfo=UTC())
	# here is what the verification logs said about early clocks:
	#~ h30gl_Sm_089-0747: APHIDS clock is early by -2.267578125 microseconds
	#~ h30gl_Sm_089-0757: APHIDS clock is early by 9.6298828125 microseconds
	#~ h30gl_Sm_089-0810: APHIDS clock is early by 3.6806640625 microseconds
	#~ h30gl_Sm_089-0817: APHIDS clock is early by 0.603515625 microseconds
	#~ h30gl_Sm_089-0827: APHIDS clock is early by 4.50146484375 microseconds
	#~ h30gl_Sm_089-0840: APHIDS clock is early by 6.552734375 microseconds
	#~ h30gl_Sm_089-0847: APHIDS clock is early by -4.52392578125 microseconds
	#~ h30gl_Sm_089-0857: APHIDS clock is early by 7.37353515625 microseconds
	#~ h30gl_Sm_089-0920: APHIDS clock is early by 13.3217773438 microseconds
	clks = ClockDef.from_csv_file(start,stop,filename='./delays.csv')
	print vex_block(clks)
	
	# now let's add another known delay 
	for clk in clks:
		clk.add_delay(-10.0)
	print vex_block(clks)
	
	# and for day2_Sm_212-0950
	start = datetime(2015,7,31,9,45,0,tzinfo=UTC())
	stop = datetime(2015,7,31,9,58,0,tzinfo=UTC())
	# there is only one scan for within this time range
	#~ APHIDS clock is early by 8.92333984375 microseconds
	clks = ClockDef.from_csv_file(start,stop,filename='./delays.csv')
	print vex_block(clks,station_id="Sn")

if __name__ == "__main__":
	from argparse import ArgumentParser,RawTextHelpFormatter
	from sys import argv
	
	# parse input arguments
	parser = ArgumentParser(description=
		"Print clock early definitions in VEX format to stdout for given start and stop range.",
							epilog=
		"Examples:\n"
		"  To print clock definitions over the range 08h00-09h00 on day 89 use\n"
		"    $ ./{0} 2015y089d08h00m00s 2015y089d09h00m00s\n"
		"  To do the same but make the clocks early by a further half microsecond\n"
		"    $ ./{0} -c 0.5 2015y089d08h00m00s 2015y089d09h00m00s\n"
		"  To specify a rate of 1.5 picoseconds per second\n"
		"    $ ./{0} -r 1.5e-12 2015y089d08h00m00s 2015y089d09h00m00s\n".format(argv[0]) +
		"  To use delays from file DELAYSFILE\n"
		"    $ ./{0} -f DELAYSFILE 2015y089d08h00m00s 2015y089d09h00m00s".format(argv[0]),
		formatter_class=RawTextHelpFormatter
	)
	parser.add_argument("-c", "--clk-offset", metavar="CLKOFFSET", type=float, default=0.0,
						help="offset to add to clock as float in microseconds, positive means towards early")
	parser.add_argument("-f", "--delays-file", metavar="DELAYSFILE", type=str, default="delays.csv",
						help="specify name of file that stores clock information in csv-like format (default is 'delays.csv')")
	parser.add_argument("-r", "--rate", metavar="RATE", type=float, default=0.0,
						help="rate specification as float in seconds per second")
	parser.add_argument("-s", "--station-id", metavar="STATIONID", type=str, default="Sm",
						help="two-character station ID (default is 'Sm')")
	parser.add_argument("start", metavar="START", type=str,
						help="start of time range in format {0}".format(vex_time_fmt.replace("%","%%")))
	parser.add_argument("stop", metavar="STOP", type=str,
						help="start of time range in format {0}".format(vex_time_fmt.replace("%","%%")))
	args = parser.parse_args()
	
	start = datetime.strptime(args.start,vex_time_fmt).replace(tzinfo=UTC())
	stop = datetime.strptime(args.stop,vex_time_fmt).replace(tzinfo=UTC())
	clks = ClockDef.from_csv_file(start,stop,filename=args.delays_file,offset=args.clk_offset,rate=args.rate)
	#~ for clk in clks:
		#~ clk.add_delay(-1.0*args.clk_offset)
	print vex_block(clks,station_id=args.station_id)
