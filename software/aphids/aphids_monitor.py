#!/usr/bin/env python

from curses import wrapper, curs_set, A_STANDOUT
from redis import StrictRedis

def main(stdscr):

    # no delay on user input
    stdscr.nodelay(True)

    # move cursos to 0, 0
    stdscr.leaveok(0)

    # connect to the redis server
    r = StrictRedis(host='127.0.0.1', port=6379, db=0)

    # main loop
    while True:

        # clear screen
        stdscr.erase()

        # print info message
        stdscr.addstr(0, 0, "press 'q' to exit.")

        # get list of aphids keys
        akeys = sorted(r.scan_iter('aphids*'))

        # get and show all aphids keys
        for i, k in enumerate(akeys):

            # get the type of this key
            k_type = r.type(k)

            # read differently depending on type
            if k_type == "string":
                v = r.get(k)
            elif k_type == "list":
                v = r.lrange(k, 0, -1)
            opts = A_STANDOUT if k.endswith('seg_rate') else 0
            stdscr.addstr(i+2, 8, k)
            stdscr.addstr(i+2, 68, "{0}".format(v), opts)

        # refresh screen
        stdscr.refresh()

        # handle user input
        try:
            c = stdscr.getkey()
            if c == 'q':
                break
        except:
            continue

wrapper(main)
