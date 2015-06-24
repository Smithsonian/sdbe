#!/usr/bin/env python

from os import kill, remove
from signal import SIGTERM

if __name__ == "__main__":

    import sys

    ID = int(sys.argv[1])
    PIDFILENAME = "/tmp/aphids.{0}.pid".format(ID)

    with open(PIDFILENAME, "r") as pidfile:
        pid = int(pidfile.readline())

    kill(pid, SIGTERM)

    remove(PIDFILENAME)
