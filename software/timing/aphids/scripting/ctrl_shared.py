#!/usr/bin/env python

PREFIX = "Ctrl"
SEPARATOR = ":"

# channel names
RX = "RX"
APHIDS = "APHIDS"
TX = "TX"
GLOBAL = "GLOBAL"

# publish messages
MSG_START = "start"
MSG_STOP = "stop"

# key names
PROCESS_NAME = "ProcessName"

# all control channels and keys get the same prefix
CHAN_RX = PREFIX + SEPARATOR + RX
CHAN_APHIDS = PREFIX + SEPARATOR + APHIDS
CHAN_TX = PREFIX + SEPARATOR + TX
CHAN_GLOBAL = PREFIX + SEPARATOR + GLOBAL
KEY_PROCESS_NAME = PREFIX + SEPARATOR + PROCESS_NAME

# use port-forwarding on hamster so connecting to localhost:defaultport
# will get us there
REDIS_HOST = "localhost"
REDIS_PORT = "6379"
