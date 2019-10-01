#!/usr/bin/env python3

import time
import sys


class Timer(object):

    def __init__(self, verbose=False, stream=sys.stderr, message=""):
        self.verbose = (verbose or (message != ""))
        self.stream = stream
        self.base_message = message

    def __enter__(self):
        self.start = time.time()
        self.lap_t = self.start
        return self

    def lap(self, message=""):
        now = time.time()
        msecs = 1000*(now - self.lap_t)
        self.lap_t = now
        msg = "" if message == "" else message + " "
        print("{}lap time: {:f} ms".format(msg, msecs), file=self.stream)

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            prefix = ("Elapsed time" if self.base_message == "" else
                      self.base_message)
            message =  "{} took {:f} ms".format(prefix, self.msecs)
            print(message, file=self.stream)
