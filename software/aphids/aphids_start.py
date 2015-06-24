#!/usr/bin/env python

from subprocess import Popen

CPU_IN = 1
CPU_INOUT = 2
CPU_OUT = 3

HASHPIPE_CMD = "hashpipe -I {0} -p aphids -c {1} vdif_in_null_thread -c ${2} vdif_inout_null_thread -c {3} vdif_out_null_thread"

if __name__ == "__main__":

    import sys

    ID = int(sys.argv[1])

    with open("stdout.log", "w") as stdout, open("stderr.log", "w") as stderr:
        formatted_cmd = HASHPIPE_CMD.format(ID, CPU_IN, CPU_INOUT, CPU_OUT)
        process = Popen(formatted_cmd.split(), stdout=stdout, stderr=stderr)

    with open("/tmp/aphids.{0}.pid".format(ID), "w") as pidfile:
        pidfile.write("{0}".format(process.pid))
