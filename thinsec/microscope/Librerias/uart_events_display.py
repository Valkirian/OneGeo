#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import signal
import sys

import zmq


def main():

    zmq_ctx = zmq.Context()
    socket = zmq_ctx.socket(zmq.SUB)

    socket.connect("ipc://uart")
    socket.setsockopt(zmq.SUBSCRIBE, b'')

    def shutdown(signum, frame):

        print("\nShutting down...")
        zmq_ctx.destroy()
        sys.exit()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        print socket.recv(),


if __name__ == "__main__":
    main()
