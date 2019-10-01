# -*- coding: utf-8 -*-

import gevent.monkey
gevent.monkey.patch_all()

import argparse
from collections import OrderedDict
import json
import signal
import sys

from flask import Flask, send_file
import gevent
from geventwebsocket import (WebSocketServer, WebSocketApplication, Resource)
import zmq.green as zmq

from common import (DebugLog, get_utc_now_ms)

debug_log = DebugLog()

websockets = set()
objects = dict()


def main():

    opt = process_command_line()
    print opt

    zmq_ctx = zmq.Context()

    resources = OrderedDict()
    resources['^/websocket'] = WSApplication
    resources['^/.*'] = app
    server = WebSocketServer(('0.0.0.0', opt.port), Resource(resources))

    def shutdown():

        print("\nShutting down...")

        server.stop()
        zmq_ctx.destroy()
        sys.exit()

    gevent.signal(signal.SIGINT, shutdown)
    gevent.signal(signal.SIGTERM, shutdown)

    server.serve_forever()


# The webserver object
app = Flask("main_server")


@app.route('/')
def root():
    return app.send_static_file('cam.html')


@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/last_image')
def last_image():
    stream = objects['last-image-data']
    debug_log("Will send an image")
    stream.seek(0)
    return send_file(stream, mimetype='image/png')


# Websockets server implementation
class WSApplication(WebSocketApplication):

    def on_open(self):

        websockets.add(self.ws)
        debug_log(repr(self.ws), "connected.")

    def on_close(self, reason):

        websockets.discard(self.ws)
        debug_log(repr(self.ws), "disconnected.")

    def on_message(self, message):

        if message is None: return

        msg = json.loads(message)
        message_type = msg['type']
        entity_name = msg['name']


def websockets_broadcast(data, identity, kind):

    for ws in websockets:
        send_to_client(ws, identity, data, kind)


def send_to_client(websocket, name, payload, message_type):

    data = {'type': message_type,
            'time': get_utc_now_ms(),
            'name': name,
            'data': payload}
    websocket.send(json.dumps(data))


def camera_status_callback(data, identity):

    controller = objects['swp-ctl']
    controller.camera_event_dispatch(data)
    websockets_broadcast(data, identity, "stt")


def sweep_status_broadcast(sweep_event):

    if sweep_event == "start":
        report = {'complete': False}
    elif sweep_event == "done":
        report = {'complete': True}
    else:
        angle, row, col = map(int, sweep_event.split(','))
        report = {'angle': angle, 'row': row, 'col': col}

    websockets_broadcast(report, 'sweep', 'stt')


def process_command_line():

    description = "Web UI interface server for tests"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("port", type=int, default=5000,
                help="TCP port over which to listen")

    return parser.parse_args()


if __name__ == "__main__":
    main()
