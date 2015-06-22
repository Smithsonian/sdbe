#!/usr/bin/env python

from curses import wrapper, curs_set
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
            v = r.get(k)
            stdscr.addstr(i+2, 8, '{0:80s}= {1}'.format(k, v))

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
