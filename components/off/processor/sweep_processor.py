#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import bz2
import json
import os
import select
import signal
import shutil
import socket
import subprocess
import sys
import tempfile

import msgpack
import zmq

from common import (DebugLog, log_line, spawn_subdir)
from LibStitch import (stitch_as_row, stitch_as_column)

debug_log = DebugLog()

job_types = ('row', 'col', 'image-by-rows', 'image-by-blocks')


def main():

    zmq_ctx = zmq.Context()
    cancellable = False
    this_id = "{}-{}".format(socket.getfqdn(), os.getpid())

    def signal_handler(signum, frame):

        print("\nShutting down...")

        notify_socket.send_json([this_id, 'del'])
        if cancellable:
            work_socket.send(msgpack.packb({'success': False, 'retry': True}),
                             flags=zmq.NOBLOCK)
        zmq_ctx.destroy()
        sys.exit()

    signal.signal(signal.SIGINT, signal_handler)

    opt = process_command_line()
    print opt

    work_socket = zmq_ctx.socket(zmq.REP)
    notify_socket = zmq_ctx.socket(zmq.PUSH)

    host, _port = opt.router_address.strip().split(':')
    port = int(_port)
    work_socket.connect("tcp://{}:{:d}".format(host, port))
    notify_socket.connect("tcp://{}:{:d}".format(host, opt.notify_port))

    print("Node", this_id, "started.")

    while True:

        notify_while_no_work(notify_socket, work_socket, this_id,
                             opt.notify_period)

        job = msgpack.unpackb(work_socket.recv())
        notify_socket.send_json([this_id, 'del'])

        work_dir = tempfile.mkdtemp(dir=opt.work_dir)
        work_dirs = setup_directories(work_dir)
        debug_log("JOB", "Staging files in", work_dir)

        cancellable = True
        job_type = job['type']

        if job_type not in job_types:
            debug_log("\nERROR", "Job type '{}' is not one of {}".format(job_type,
                                                                         ', '.join(job_types)))
            blob = msgpack.packb({'success': False, 'data': None, 'logs': None})
            work_socket.send(blob)
            continue

        try:
            cancellable = False

            if job_type == 'row':
                reply = assemble_row(work_dirs, job['data'],
                                     job['id'], job['dimensions'])
            elif job_type == 'col':
                reply = assemble_column(work_dirs, job['data'],
                                        job['id'], job['dimensions'])

            blob = msgpack.packb(reply)
            work_socket.send(blob)
            debug_log("INFO", "Sent reply blob", len(blob), "B")

        finally:
            shutil.rmtree(work_dir)


def assemble_row(directories, data, identity, dimensions):

    return perform_assembly(directories, data, dimensions,
                            "row", stitch_as_row, identity)


def assemble_column(directories, data, identity, dimensions):

    return perform_assembly(directories, data, dimensions,
                            "column", stitch_as_column, identity)


def perform_assembly(directories, data, dimensions, kind_tag,
                     assembly_fun, identity):

    exec_fun = command_executor_factory(directories['log'], kind_tag)

    debug_log("Assembling", kind_tag, identity, "on", directories['in'],
              "started")

    files = []
    for pic, jpeg in data.iteritems():
        pic_path = os.path.join(directories['in'], pic)
        with open(pic_path, 'w') as img:
            img.write(jpeg)
        files.append(pic_path)

    out_file, success = assembly_fun(exec_fun, dimensions,
                                     directories['in'], directories['out'])

    reply = {'success': success,
             'logs': collect_text_files(directories['log'])}

    if success:
        reply['data'] = open(out_file).read()

    result_name = "done" if success else "failed"
    debug_log(kind_tag, "Assembling on", directories['in'], result_name)

    return reply


def collect_text_files(directory):

    collection = {}

    for filenom in os.listdir(directory):
        file_path = os.path.join(directory, filenom)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fobj:
                collection[filenom] = fobj.read()

    return bz2.compress(json.dumps(collection))


def setup_directories(work_dir):

    temp_dir = spawn_subdir(work_dir, "temp")
    in_dir = spawn_subdir(work_dir, "input")
    out_dir = spawn_subdir(work_dir, "output")
    log_dir = spawn_subdir(work_dir, "log")
    dirs = {'temp': temp_dir, 'out': out_dir, 'in': in_dir, 'log': log_dir}

    return dirs


READ_ONLY = select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR

class command_executor_factory(object):

    def __init__(self, log_dir, log_name_base, print_to_stdout=True):

        self.log_dir = log_dir
        self.log_name_base = log_name_base
        self.print_to_stdout = print_to_stdout

    def __call__(self, command):

        start_line = log_line("Invoking", command)
        proc = subprocess.Popen(command, shell=True, bufsize=1,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        stdout_l = []
        stderr_l = []
        poller = select.poll()
        poller.register(proc.stdout, READ_ONLY)
        poller.register(proc.stderr, READ_ONLY)
        fd_to_file = {proc.stdout.fileno(): proc.stdout,
                      proc.stderr.fileno(): proc.stderr}
        while proc.poll() is None:
            events = poller.poll(100)
            for fd, flag in events:
                file_o = fd_to_file[fd]
                if file_o is proc.stdout:
                    out = proc.stdout.readline().rstrip()
                    if out:
                        stdout_l.append(out)
                        if self.print_to_stdout:
                            print("{} stdout: {}".format(self.log_name_base, out))
                elif file_o is proc.stderr:
                    err = proc.stderr.readline().rstrip()
                    if err:
                        stderr_l.append(err)
                        if self.print_to_stdout:
                            print("{} stderr: {}".format(self.log_name_base, err))
        stdout = '\n'.join(stdout_l)
        stderr = '\n'.join(stderr_l)

        retcode = proc.returncode
        end_line = log_line("Finished", command, "with retcode", retcode)

        for data, name in zip((stdout, stderr), ('out', 'err')):

            log_filename = "{}.{}".format(self.log_name_base, name)
            log_path = os.path.join(self.log_dir, log_filename)
            with open(log_path, 'a') as fobj:
                fobj.write('='*80 + '\n')
                fobj.write(start_line + '\n')
                fobj.write(data + '\n')
                fobj.write(end_line + '\n')

        return retcode


def notify_while_no_work(notify_socket, work_socket, my_id, notify_period):

    while True:
        notify_socket.send_json([my_id, 'add'])
        if work_socket.poll(int(notify_period*1000)):
            return


def process_command_line():

    description = ("Performs stitching operations on a set of files as "
                   "dictated by the Dealer")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-r', "--router-address", default='localhost:6000',
            help="Address over which the router transfers work from the dealer")
    parser.add_argument('-n', "--notify-port", type=int, default=60000,
            help=("TCP port to which the dealer listens for notifications"
                  " from processing nodes"))
    parser.add_argument('-d', "--work-dir", default='/dev/shm',
            help="Where input, temporary, and output files will be written")
    parser.add_argument('-p', "--notify-period", type=float, default=0.5,
            help=("Period in seconds of availability notification for "
                  "the dealer"))

    return parser.parse_args()


if __name__ == "__main__":
    main()
