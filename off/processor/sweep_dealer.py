#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import bz2
import collections
from functools import partial
import json
import os
import signal
import sys

from fysom import Fysom
import gevent
import msgpack
import zmq.green as zmq

from common import (DebugLog, print_state_change, spawn_subdir,
                    image_array_crop, img_data_fmt, get_utc_now_ms)
from cv_tools import(stream_to_cv2, cv2_to_file, image_resize,
                     blankfield_linear_correct)

debug_log = DebugLog()


def main():

    zmq_ctx = zmq.Context()

    def signal_handler(signum, frame):
        print("\nShutting down...")
        zmq_ctx.destroy()
        sys.exit()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    opt = process_command_line()
    print opt

    ctl_socket = zmq_ctx.socket(zmq.REP)
    img_socket = zmq_ctx.socket(zmq.REQ)
    svc_socket = zmq_ctx.socket(zmq.PULL)

    host, _port = opt.server_address.strip().split(':')
    port = int(_port)
    ctl_socket.connect("tcp://{}:{:d}".format(host, port))
    img_socket.connect("tcp://{}:{:d}".format(host, port + 1))
    svc_socket.bind("tcp://*:{:d}".format(opt.service_update_port))

    nodes_available = set()
    sweeps = {}
    work_queue = collections.deque()

    poller = zmq.Poller()
    poller.register(ctl_socket, zmq.POLLIN)
    poller.register(svc_socket, zmq.POLLIN)

    while True:

        socks = dict(poller.poll())

        if ctl_socket in socks:

            command, data = msgpack.unpackb(ctl_socket.recv())
            debug_log("RUN", data['id'], command)
            sweep_id = data['id']

            if command == "start":
                debug_log("RUN", {key: val for key, val in data.items()
                                    if key != 'blank-field'})
                enqueue_fun = partial(enqueue_stitch_job, sweeps=sweeps,
                                      work_queue=work_queue,
                                      sweep_id=sweep_id,
                                      nodes_available=nodes_available)
                dealer = SweepDealer(sweep_id, zmq_ctx, img_socket, enqueue_fun,
                                     opt.output_dir, opt.router_addr,
                                     opt.reduction_percentage)
                sweeps[sweep_id] = dealer
                sweeps[sweep_id].fsm.start(data)
                ctl_reply = 'OK'

            elif command == "stop":
                sweeps[sweep_id].fsm.stop()
                ctl_reply = 'OK'

            elif command == "new_pic":
                debug_log("RUN", data)
                if not sweeps[sweep_id].fsm.isstate('idle'):
                    sweeps[sweep_id].save_picture(data['filename'],
                                                  data['position'],
                                                  data['crop'])
                    ctl_reply = 'OK'
                else:
                    debug_log("ERR", "Picture received without starting sweep")
                    ctl_reply = 'ERROR'

            ctl_socket.send(ctl_reply)

        if svc_socket in socks:
            get_notifications(svc_socket, nodes_available, work_queue, sweeps)


class SweepDealer(object):

    def __init__(self, my_id, zmq_context, img_socket, queueing_fun,
                 output_dir, router_address, resize_factor=100, debug=False):

        self.my_id = my_id
        self.zmq_context = zmq_context
        self.socket_img = img_socket
        self.enqueue_fun = queueing_fun
        self.output_dir = output_dir
        self.router_address = router_address
        self.resize_ratio = resize_factor

        events = [{'name': 'start', 'src': 'idle', 'dst': 'in_sweep'},
                  {'name': 'stop', 'src': '*', 'dst': 'idle'}]
        callbacks = {'onleaveidle': self.reset,
                     'onstop': self.halt}
        self.fsm = Fysom(initial='idle', events=events, callbacks=callbacks)

        self.time_start = get_utc_now_ms()
        if debug:
            self.fsm.onchangestate = print_state_change

    def reset(self, event):

        params = event.args[0]
        self.angles = params['pol-angle']
        self.rows = params['nrow']
        self.cols = params['ncol']
        self.strides = params['strides']
        self.focus = params['focus']
        self.project_timestamp = params['timestamp']
        self.project_name = params['cname']
        self.planar_dimensions = {'cols': self.cols, 'rows': self.rows}
        self.set_blank_field(params['blank-field'])

        full_name = "{}-{}".format(self.project_timestamp, self.project_name)
        self.project_dir = spawn_subdir(self.output_dir, full_name)

        self.cam_dirs = {}
        self.parts_table = []

        cam_name = 'cam0'
        cam_dir = spawn_subdir(self.project_dir, cam_name)
        angles_dir = spawn_subdir(cam_dir, "angles")
        self.cam_dirs[cam_name] = {'base': cam_dir,
                                   'angles': {}}
        for angle in range(self.angles):

            angle_dir = spawn_subdir(angles_dir, str(angle))
            cells_dir = spawn_subdir(angle_dir, "cells")
            rows_dir = spawn_subdir(angle_dir, "rows")
            log_dir = spawn_subdir(angle_dir, "log")
            self.cam_dirs[cam_name]['angles'][angle] = {'base': angle_dir,
                                                        'cells': cells_dir,
                                                        'cell_files': {},
                                                        'rows': rows_dir,
                                                        'row_files': [],
                                                        'log': log_dir}
            self.parts_table.append({'rows': {row: False for row
                                               in range(self.rows)},
                                     'image': False})
            settings_file_path = os.path.join(angle_dir, "settings.json")
            with open(settings_file_path, 'w') as sf:
                json.dump({'bf-corrected': False}, sf)
                #json.dump({'bf-corrected': (self.blank_field is None)}, sf)

        focus_file = os.path.join(self.project_dir, 'focus.json')
        with open(focus_file, 'w') as fobj:
            json.dump(self.focus, fobj)

        debug_log("FSM", "Starting sweep...", self.rows, self.cols, self.angles)

    def halt(self, event):

        elapsed_ms = get_utc_now_ms() - self.time_start
        debug_log("FSM", "Sweep terminated (took", elapsed_ms/60e3, "min)")

    def save_picture(self, base_filename, positions, crop):

        angle = positions['ana']
        row = positions['y']

        self.socket_img.send_json(["send_pics", {'positions': positions}])
        blob = msgpack.unpackb(self.socket_img.recv())
        for name, pics_data in blob.items():

            logical_step = pics_data['step']
            still_data = pics_data['data']

            debug_log("POS", base_filename,
                      "step:", sorted(logical_step.items()))

            this_angle = self.cam_dirs[name]['angles'][angle]
            directory = this_angle['cells']
            still_path = os.path.join(directory, base_filename)
            uncorrected_still_path = os.path.join(directory,
                                                  base_filename)
            blank_field_path = os.path.join(directory,
                                            "blank_field.jpg")
            if not os.path.exists(blank_field_path):
                with open(blank_field_path, 'w') as fobj:
                    fobj.write(self.blank_field_data)

            gevent.spawn(self.preprocess_image, still_path,
                         uncorrected_still_path, still_data,
                         crop, angle, row, this_angle, logical_step, name)

    def preprocess_image(self, still_path, uncorrected_still_path, still_data,
                         crop, angle, row, this_angle, logical_step, name):

        image = stream_to_cv2(still_data)

        img = image_array_crop(image, crop) if crop != [0, 0, 1, 1] else image
        """
        img_r = image_resize(img, self.resize_ratio)

        if self.blank_field is not None:
            bkg = ( image_array_crop(self.blank_field, crop)
                    if crop != [0, 0, 1, 1] else self.blank_field )
            bkg_r = image_resize(bkg, self.resize_ratio)

            cv2_to_file(blankfield_linear_correct(img_r, bkg_r), still_path, 95)
            #blankfield_divide_correct(img_r, bkg_r, still_path)
        """

        cv2_to_file(img, uncorrected_still_path, 95)
        written = os.path.getsize(uncorrected_still_path)
        debug_log("SvP", "wrote", written/1024, "kiB to", uncorrected_still_path)

        if row not in this_angle['cell_files']:
            this_angle['cell_files'][row] = []
        this_angle['cell_files'][row].append(still_path)

        """
        end_of_row = logical_step['x'] == self.cols - 1
        if end_of_row:
            self.enqueue_fun("row", row, name, angle)
        """

    def execute_row_stitching(self, camera, angle, row):

        job_socket = self.zmq_context.socket(zmq.REQ)
        job_socket.connect(self.router_address)

        this_angle = self.cam_dirs[camera]['angles'][angle]
        files = this_angle['cell_files'][row]
        blob = { os.path.basename(pic): open(pic).read() for pic in files }
        job = msgpack.packb({'type': 'row',
                             'data': blob,
                             'id': row,
                             'dimensions': self.planar_dimensions})
        job_socket.send(job)
        debug_log("Job", "Sent row", row, "for stitching",
                  "(payload = {:.1f} MiB)".format(len(job)/2.0**20),
                  "files:\n", files)

        reply = msgpack.unpackb(job_socket.recv())
        job_socket.close()

        if reply['success']:
            row_dir = this_angle['rows']
            the_row = os.path.basename(files[0]).split('_')[0]
            row_file = os.path.join(row_dir, the_row + "." + img_data_fmt)
            row_data = reply['data']
            with open(row_file, 'w') as fobj:
                fobj.write(row_data)
            debug_log("Job", "Succesfully saved row", row,
                      "(size = {:.1f} MiB)".format(len(row_data)/2.0**20))
            this_angle['row_files'].append(row_file)
        else:
            debug_log("Job:", "Failed stitching row", row, "of angle", angle)

        write_logs(reply['logs'], this_angle['log'], row)

        self.parts_table[angle]['rows'][row] = reply['success']

        if all(self.parts_table[angle]['rows'].values()):
            self.enqueue_fun("col", None, camera, angle)

    def execute_column_stitching(self, camera, angle):

        job_socket = self.zmq_context.socket(zmq.REQ)
        job_socket.connect(self.router_address)

        this_angle = self.cam_dirs[camera]['angles'][angle]
        files = this_angle['row_files']
        blob = { os.path.basename(pic): open(pic).read() for pic in files }
        job = msgpack.packb({'type': 'col',
                             'data': blob,
                             'id': angle,
                             'dimensions': self.planar_dimensions})
        job_socket.send(job)
        debug_log("Job", "Sent rows of angle", angle, "for stitching",
                  "(payload = {:.1f} MiB)".format(len(job)/2.0**20))

        reply = msgpack.unpackb(job_socket.recv())
        job_socket.close()

        if reply['success']:
            img_dir = this_angle['base']
            img_file = os.path.join(img_dir, "assembly." + img_data_fmt)
            img_data = reply['data']
            with open(img_file, 'w') as fobj:
                fobj.write(img_data)
            debug_log("Job", "Succesfully saved angle image", angle,
                      "(size = {:.1f} MiB)".format(len(img_data)/2**20))
        else:
            debug_log("Job", "Failed stitching angle", angle)

        logs = json.loads(bz2.decompress(reply['logs']))
        log_dir = this_angle['log']
        for logname, logdata in logs.iteritems():
            logfile = os.path.join(log_dir, logname)
            with open(logfile, 'w') as logfi:
                logfi.write(logdata)

        self.parts_table[angle]['image'] = reply['success']

    def set_blank_field(self, data):

        """
        self.blank_field = (precondition_blankfield(stream_to_cv2(data))
                            if data else None)
        self.blank_field_data = (cv2_to_stream(self.blank_field, quality=99)
                                 if self.blank_field is not None else '')
        """
        self.blank_field_data = data
        self.blank_field = stream_to_cv2(data) if data != '' else None


def get_notifications(service_socket, nodes_available, work_queue, sweeps):

    worker, action = service_socket.recv_json()

    if action == 'add' and worker not in nodes_available:
        debug_log("PRO", "Node", worker, "added")
        nodes_available.add(worker)
        poll_for_work(nodes_available, work_queue, sweeps)

    elif action == 'del' and worker in nodes_available:
        debug_log("PRO", "Node", worker, "removed")
        nodes_available.discard(worker)


def poll_for_work(nodes_available, work_queue, sweeps):

    if len(work_queue) > 0 and len(nodes_available) > 0:

        sweep_id, cam, angle, job_type, job_id = work_queue.popleft()
        dealer = sweeps[sweep_id]

        if job_type == "row":
            gevent.spawn(dealer.execute_row_stitching, cam, angle, job_id)
        elif job_type == "image":
            gevent.spawn(dealer.execute_column_stitching, cam, angle)


def enqueue_stitch_job(job_type, job_id, camera, angle,
                       work_queue, nodes_available, sweeps, sweep_id):

    work_queue.append((sweep_id, camera, angle, job_type, job_id))
    poll_for_work(nodes_available, work_queue, sweeps)


def write_logs(log_blob, log_dir, row):

    if log_blob is None:
        return

    logs = json.loads(bz2.decompress(log_blob))
    for logname, logdata in logs.iteritems():
        part_name, part_ext = logname.split('.')
        new_name = "{}_{:d}.{}".format(part_name, row, part_ext)
        logfile = os.path.join(log_dir, new_name)
        with open(logfile, 'w') as logfi:
            logfi.write(logdata)


def process_command_line():

    description = "Controls the two-camera photo sweep via ZeroMQ"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-a', "--server-address", default='localhost:6000',
            help="Base port over which ZeroMQ-based services are listening")
    parser.add_argument('-r', "--router-addr", default="ipc:///tmp/transport",
            help=("<protocol>://<address> string for any of the transport "
                  "methods supported by ZeroMQ"))
    parser.add_argument('-u', "--service-update-port", type=int, default=50000,
            help=("TCP port number to listen to for receiving notifications "
                  "from processing nodes"))
    parser.add_argument('-o', "--output-dir", default='/dev/shm',
            help="Where the cells, rows and assembly images will be stored")
    parser.add_argument('-z', "--reduction-percentage", type=int, default=100,
            help=("Percentage of dimension span to scale the images to"))

    return parser.parse_args()


if __name__ == "__main__":
    main()
