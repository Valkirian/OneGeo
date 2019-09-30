#!/usr/bin/env python2

import json
import gevent
import numpy as np
import zmq.green as zmq

from common import DebugLog
debug_log = DebugLog()


class CameraLink(object):

    def __init__(self, zmq_context, name):

        self.name = name

        # pusher that emits camera configuration commands
        self.cmd_push = zmq_context.socket(zmq.PUSH)

        # subscriber that receives status updates
        self.inf_subs_stt = zmq_context.socket(zmq.SUB)
        self.inf_subs_stt.setsockopt(zmq.SUBSCRIBE, b'_stt_')
        # subscriber that receives configuration data
        self.inf_subs_cfg = zmq_context.socket(zmq.SUB)
        self.inf_subs_cfg.setsockopt(zmq.SUBSCRIBE, b'_cfg_')

        # subscriber that receives video frames
        self.img_subs = zmq_context.socket(zmq.SUB)
        self.img_subs.setsockopt(zmq.SUBSCRIBE, b'')
        self.last_frame_data = ''

    def connect(self, base_address):

        cmd_addr, stt_addr, vid_addr = get_socket_adresses(base_address)

        self.cmd_push.connect(cmd_addr)
        self.inf_subs_stt.connect(stt_addr)
        self.inf_subs_cfg.connect(stt_addr)
        self.img_subs.connect(vid_addr)

    def start(self, callback_config, callback_status):

        gevent.spawn(self.report_camera_config, callback_config)
        gevent.spawn(self.report_camera_status, callback_status)
        gevent.spawn(self.fetch_frames)

    def report_camera_config(self, callback):

        while True:
            rec = self.inf_subs_cfg.recv()
            message = json.loads(rec[5:])
            callback(message)

    def report_camera_status(self, callback):

        while True:
            rec = self.inf_subs_stt.recv()
            message = json.loads(rec[5:])
            callback(message)

    def fetch_frames(self):

        while True:
            self.last_frame_data = self.img_subs.recv()

    def take_picture(self):

        cols, rows = 2448, 2048
        frame = np.fromstring(self.last_frame_data, np.uint8)
        frame.resize((rows, cols, 3))

        return frame

    def set_config(self, params):

        self.cmd_push_dbg('config', params)

    def set_power(self, on=True):

        self.cmd_push_dbg('state', {'power': on})

    def set_sweep_settings(self, settings):

        self.cmd_push_dbg('sweep', settings)

    def configure_fastaxis_scan(self, settings):

        self.cmd_push_dbg('fastaxis', settings)

    def cmd_push_dbg(self, name, params):

        debug_log("->", "CMD", name, params)
        self.cmd_push.send_json([name, params])


def get_socket_adresses(base_address):

    if base_address.startswith("ipc://"):
        cmd_addr = base_address + ".cmd"
        stt_addr = base_address + ".stt"
        vid_addr = base_address + ".vid"

    elif base_address.startswith("tcp://"):
        host, port_s = base_address.split(':')
        port = int(port_s)
        cmd_addr = "{}:{:d}".format(host, port)
        stt_addr = "{}:{:d}".format(host, port + 1)
        vid_addr = "{}:{:d}".format(host, port + 2)

    else:
        raise ValueError("Unsupported transport type")

    return cmd_addr, stt_addr, vid_addr
