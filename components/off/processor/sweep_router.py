#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import signal
import sys

import zmq


def main():

    context = zmq.Context()

    def signal_handler(signum, frame):

        print("\nShutting down...")
        context.destroy()
        sys.exit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    opt = process_command_line()
    print opt

    # Socket facing clients
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(opt.address)
    # Socket facing services
    backend = context.socket(zmq.DEALER)
    backend.bind("tcp://*:{}".format(opt.stitch_port))

    zmq.device(zmq.QUEUE, frontend, backend)


def process_command_line():

    description = "Distributes stitching workloads among available processors"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-a', "--address", default='ipc:///tmp/transport',
            help=("<protocol>://<address> string for any of the transport "
                  "methods supported by ZeroMQ"))
    parser.add_argument('-p', "--stitch-port", type=int, default=60000,
            help=("TCP port number to listen to for serving workloads from "
                  "the dealer"))

    return parser.parse_args()


if __name__ == "__main__":
    main()
