# -*- coding: utf-8 -*-

# Importando todas las librerias necesarias
# Parchando modulos
from gevent import monkey
monkey.patch_all()

import argparse
from collections import OrderedDict
import functools
import json
import signal
import struct
from io import StringIO
import sys
import cv2
from flask import Flask, send_file
import gevent
from geventwebsocket import WebSocketServer, WebSocketApplication, Resource
import zmq.green as zmq


# Importando funciones de otros archivos
# Importando la funcion para cerrar la ejecucion y el parser

import process_line
from camera_cv_tools import CameraLink
from camera_cv_tools import color_to_gray
from common import DebugLog, get_utc_now_ms
import dft_stitch as dft
import motor_api
from motor_uart import MotorLink
from sweep_controller import SweepController

"""
=============================
      Variables Globales
=============================
"""


debug_loggin = DebugLog()

# Llaves son los websockets y los valores son los tipos de cliente
websockets = dict()

# Guardando los objetos globales
objects = dict()
# Para este uso del metodo MotorLink, debemos ver que TTY representa el microscopio y el otro argumento son los baudios
motor_execution = MotorLink("Mirar que TTY usa en el sistema", 250000)

# Esto se tiene que cambiar cuando se quite el sh
path_to_download_image = '/image'

"""
============================
        Codigo Valioso
============================
"""


def main():
    """
    ========================================
        Codigo que no se para que sirve
    ========================================
    """

    opt = process_command_line()

    zmq_ctx = zmq.Context()

    # Pasandole el contexto del mensajero, y el nombre de la camara
    came_0 = CameraLink(zmq_ctx, 'cam0')
    came_0.connect(opt.camera_server_address)
    # Si hay mas camaras, se deben seguir asociando con este formato
    camaras = {'cam0': came_0}
    # Actualizando el objeto global
    objects.update(camaras)
    objects['last-image'] = None

    # No existe el metodo motor_debug
    # motor_execution.debug = opt.motor_debug
    
    motor_execution.restart()
    gevent.spawn(motor_execution.uart_readout_loop)

    controlador = SweepController(camaras, motor_execution, sweep_status_broadcast, debug=True)
    objects['sweep-controller'] = controlador

    # configurando el broadcast de la camara
    came_0_config_broadcast = functools.partial(websockets_broadcast, identity=came_0.name, Kind='cfg')
    # devolviendo el estado del broadcast
    came_0_status_broadcast = functools.partial(camera_status_callback, identity=came_0.name)
    # Inicializando la camara
    came_0.start(came_0_config_broadcast, came_0_status_broadcast)

    resources = OrderedDict()
    # Agregando 2 pares al objeto OrderedDict
    resources['^/websocket'] = WSApplication
    resources['^/.*'] = app

    server = WebSocketServer(('0.0.0.0', opt.port), Resource(resources))
    
    def shutdown():
        for motor in motor_api.motor_address.keys():
            motor_execution.set_hold_force(motor, False)
            gevent.sleep(0.01)
            #motor_api.py
    motor_execution.set_axes_enabled(False)
    server.stop()
    zmq_ctx.destroy()
    sys.exit()

    gevent.signal(signal.SIGINT, shutdown)
    gevent.signal(signal.SIGTERM, shutdown)
    # Procesando una o varias solicitudes
    server.serve_forever()

# WebServer
app = Flask('main_server')

@app.route('/')
def root():
    return app.send_static_file('cam.html')

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route(path_to_download_image)
def last_image():
    stream = objects['last-image-data']
    debug_log("Will send a image")
    stream.seek(0)
    return send_file(stream, mimetype='image/png')


# Implementacion del servidor de websockets
class WSApplication(WebSocketApplication):

    def on_open(self):

        ws_type = 'manager' if len(websockets) == 0 else 'monitor'
        websockets[self.ws] = ws_type
        send_to_client(self.ws, 'motor', motor_api.get_hardware_config(), 'cfg')
        debug_log(ws_type, repr(self.ws), "connected.")
        if ws_type == 'manager':
            motor_exec.set_axes_enabled(True)

    def on_close(self, reason):

        ws_type = websockets.pop(self.ws, None)
        debug_log(ws_type, repr(self.ws), "disconnected.")

    def on_message(self, message):

        if message is None: return

        msg = json.loads(message)
        message_type = msg['type']
        entity_name = msg['name']
        ws_type = websockets[self.ws]

        if ws_type == 'manager' and message_type == 'control':
            gui_user_command(entity_name, msg['message'], self.ws)

def gui_user_command(name, payload, websocket):

    debug_log("User command to", name, ":", json.dumps(payload))
    command = payload['command']
    params = payload['params']
    controller = objects['swp-ctl']
    if "cam" in name:
        camera = objects[name]
        if command == "set_cam_params":
            camera.set_config(params)
        elif command == "take_picture":
            frame = camera.take_picture()
            if frame is not None:
                retval, buff = cv2.imencode('.png', frame,[cv2.IMWRITE_PNG_COMPRESSION, 1])
                objects['last-image-data'] = StringIO.StringIO(buff.tostring())
                send_to_client(websocket, 'cam0', {'url': download_image_url}, 'img')
            else:
                debug_log("Picture fetch unsuccessful")
            new_image = color_to_gray(frame)
            if objects['last-image'] is not None:
                disp, corr = dsl.get_phasecorr_peak(objects['last-image'],new_image)
                websockets_broadcast({'deltas': list(disp[::-1]), 'corr': corr},'delta-pix', 'clc')
            objects['last-image'] = new_image
        elif command == "fan-set":
            motor_exec.set_cam_fan_speed(params['speed'])
        elif command == "power":
            camera.set_power(params['on'])
    elif "motor" in name:
        motor = name[5:] if name != "motor" else None
        if command == "move-rel":
            motor_exec.move_motor(name, 'rel', params)
        if command == "move-abs":
            motor_exec.move_motor(name, 'abs', params)
        elif command == "modestep":
            motor_exec.set_mode(motor, params)
        elif command == "dir-invert":
            invert = bool(params['invert'])
            motor_exec.set_direction_invert(motor, invert)
        elif command == "power":
            level = bool(params['level'])
            for m in motor_api.motor_address.keys():
                motor_exec.set_hold_force(m, level)
                gevent.sleep(5e-3)
        elif command == "axis-reset":
            motor_exec.set_origin(motor)
        elif command == "go-home":
            motor_exec.fix_speed(motor, -params['speed'])
        elif command == "axes-reset":
            for m in motor_api.motor_address.keys():
                motor_exec.set_origin(m)
    elif name == "sweep":
        if command == "start":
            controller.start(params)
        elif command == "stop":
            controller.stop()
        elif command == "focus-set":
            controller.focus_set()
        elif command == "focus-reset":
            controller.focus_reset()
    elif name == "plate":
        if command == "ana":
            motor_exec.con_plate_ana(True)
        elif command == "pol":
            motor_exec.con_plate_pol(params['on'])
    elif name == "lamp":
        if command == "fan-set":
            motor_exec.set_led_fan_speed(params['speed'])
        elif command == "power":
            motor_exec.set_led_state(params['on'])
    elif name == "lamp2":
        if command == "power":
            motor_exec.set_led2_state(params['on'])
    elif name == "events":
        if command == "trigger-sequence":
            motor_exec.set_trigger_positions(params['sequence'])
        elif command == "wait-time":
            motor_exec.set_trigger_wait_time(params['time'])

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


def motor_events_loop():
    bcast_spec = ('motor', 'stt')
    while True:
        data = motor_exec.uart_info_queue.get()
        if data == 'do':
            debug_log("MOT", "all motors done")
            controller = objects['swp-ctl']
            controller.motor_done_notify()
            websockets_broadcast({'move': True}, *bcast_spec)
        elif data.startswith('m'):
            motor_idx, position = struct.unpack('<bi', data[1:])
            motor = motor_exec.indexed_ids[motor_idx + 1]
            motor_exec.position[motor] = position
            websockets_broadcast({'position': {motor: position}}, *bcast_spec)
            debug_log("MOT", "motor", motor, "at", position)


def motor_io_debug_loop():
    while True:
        data = motor_exec.uart_debug_queue.get()
        debug_log("DBG", "io", repr(data))


def sweep_status_broadcast(sweep_event):
    if isinstance(sweep_event, str):
        if sweep_event == "start":
            report = {'complete': False}
        elif sweep_event == "done":
            report = {'complete': True}
    else:
        kind, angle, row, col = sweep_event
        report = {'kind': kind, 'angle': angle, 'row': row, 'col': col}
    debug_log("SWR", report)
    websockets_broadcast(report, 'sweep', 'stt')


process-command-line.process_command_line()

if __name__ == "__main__":
    main()