
import argparse
from collections import deque
import json
import os
import signal
import sys
import tarfile
import time

import cv2
import numpy as np
import picamera.array
import zmq

# Este archivo (camera_driver) no existe.
# from camera_driver import CameraDriver

from common import DebugLog, image_array_crop, default_crop

debug_log = DebugLog()

zmq_ctx = zmq.Context()

# puller that handles incoming camera configuration commands
cmd_pull = zmq_ctx.socket(zmq.PULL)
# publisher of camera config messages
config_pub = zmq_ctx.socket(zmq.PUB)
# publisher of camera frames
video_pub = zmq_ctx.socket(zmq.PUB)
# pusher that emits status updates
stt_pub = zmq_ctx.socket(zmq.PUB)
# synchronous server of camera snapshots
picture_rep = zmq_ctx.socket(zmq.REP)


def main():
    opt = process_command_line()
    print (opt)
    cmd_pull.bind("tcp://*:%d" % opt.zeromq_port)
    config_pub.bind("tcp://*:%d" % (opt.zeromq_port + 1))
    video_pub.bind("tcp://*:%d" % (opt.zeromq_port + 2))
    stt_pub.bind("tcp://*:%d" % (opt.zeromq_port + 3))
    stt_pub.setsockopt(zmq.CONFLATE, 1)
    picture_rep.bind("tcp://*:%d" % (opt.zeromq_port + 4))
    poller = zmq.Poller()
    poller.register(cmd_pull, zmq.POLLIN)
    poller.register(picture_rep, zmq.POLLIN)
    source_dim = (opt.source_width, opt.source_height)
    output_dim = (opt.width, opt.height)
    if source_dim == output_dim:
        output_dim = None
    with CameraDriver() as camera:
        motion_detector = MotionDetector(camera, size=output_dim, report=opt.motion_report) if opt.motion_detect else None

        def do_stream():
            camera.start_recording(ZMQOutput(video_pub, opt.traffic_report), format='h264', profile='baseline', resize=output_dim, motion_output=motion_detector)
        camera.init(source_dim, opt.framerate, opt.flip_horizontal, opt.flip_vertical)
        do_stream()
        last = time.time()
        while True:
            # Report camera config
            now = time.time()
            if (now - last) >= opt.report_period:
                last += opt.report_period
                config_pub.send_json(camera.get_config())
            socks = dict(poller.poll(50))
            if cmd_pull in socks:
                handle_command(camera, do_stream, opt.pictures_dir)
            if picture_rep in socks:
                take_picture(camera, opt.pictures_dir)

    return 0


def state_pub_dbg(info):
    debug_log("<-", "STT", info)
    stt_pub.send_json(info)


class ZMQOutput(object):
    def __init__(self, zmq_socket, report=False):
        self.socket = zmq_socket
        self.report = report
        self.ts = time.time()

    def write(self, outbound):
        self.socket.send(outbound)
        if self.report:
            now = time.time()
            dt = 1000*(now - self.ts)
            self.ts = now
            rate = round(len(outbound)/dt)
            report_msg = "NeA: {} bytes in {} ms ({} kB/s)\n"
            debug_log(report_msg.format(len(outbound), round(dt), rate))


def array_encode(array, kind="jpg", quality=95):
    retval, output = cv2.imencode('.' + kind, array, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_PNG_COMPRESSION, 2])

    return output.tostring()


def handle_command(camera, stream_func, pictures_dir):
    kind, command = cmd_pull.recv_json()
    debug_log("->", "CMD", kind, json.dumps(command))
    if kind == "config":
        camera.set_config(command)
    elif kind == "sweep":
        if archive_sweep(command, pictures_dir):
            result = "archived"
        else:
            result = "error"
        state_pub_dbg({kind: result})
    elif kind == "stream":
        if command['active']:
            if not camera.recording:
                stream_func()
        else:
            if camera.recording:
                camera.stop_recording()


def take_picture(camera, pictures_dir):
    command = picture_rep.recv_json()
    if command['local']:
        output = open(os.path.join(pictures_dir, command['filename']), 'w')
    else:
        output = ZMQOutput(picture_rep)
    encoder = "jpeg"
    crop = default_crop if ('crop' not in command) else list(command['crop'])
    format = 'bgr' if crop != default_crop else encoder
    capture_output = output if format == encoder else camera.adi_capture_array
    debug_log("CAP", "name is", command['filename'], "format is", format)
    toc = time.time()
    camera.adi_capture_array.truncate(0)
    camera.capture(capture_output, format=format, use_video_port=True)

    if format == 'bgr':
        image = camera.adi_capture_array.array
        cropped = (image_array_crop(image, crop) if (crop != default_crop) else image)
        file_format = encoder.replace('e', '')
        image_roi = array_encode(cropped, file_format)
        output.write(image_roi)
    tic = time.time()
    debug_log("Shoot took {:.2f} ms".format(1000*(tic-toc)))
    if command['local']:
        picture_rep.send('taken')
    state_pub_dbg({'picture': 'taken'})


def archive_sweep(parameters, work_dir):
    sweep_name = "sweep-" + parameters['proj']
    json_name = "sweep.json"
    json_path = os.path.join(work_dir, json_name)
    #tar_path = os.path.join(work_dir, sweep_name + ".tar.bz2")
    tar_path = os.path.join(work_dir, sweep_name + ".tar")
    try:
        #pkg = tarfile.open(tar_path, 'w:bz2')
        pkg = tarfile.open(tar_path, 'w')
        for pic_name in parameters['fnames']:
            pic_path = os.path.join(work_dir, pic_name)
            target_path = os.path.join(sweep_name, pic_name)
            pkg.add(pic_path, arcname=target_path)
        json.dump(parameters, open(json_path, 'w'))
        target_path = os.path.join(sweep_name, json_name)
        pkg.add(json_path, arcname=target_path)
        pkg.close()
        for pic_name in parameters['fnames']:
            pic_path = os.path.join(work_dir, pic_name)
            os.remove(pic_path)
        os.remove(json_path)
        success = True
    except IOError:
        debug_log("Error while creating sweep archive:", e)
        success = False

    return success


class MotionDetector(picamera.array.PiMotionAnalysis):
    def __init__(self, *args, **kwargs):
        self.report = kwargs.pop('report', False)
        super(MotionDetector, self).__init__(*args, **kwargs)
        self.window = 2
        self.queue = deque(maxlen=self.window)
        self.in_motion = False

    def analyse(self, a):
        toc = time.time()
        a = np.sqrt(
            np.square(a['x'].astype(np.float)) +
            np.square(a['y'].astype(np.float))
            ).clip(0, 255).astype(np.uint8)
        if (a > 80).sum() > 1:
            self.queue.append(True)
        else:
            self.queue.append(False)
        count_true = sum(detect for detect in self.queue)
        count_false = sum(not detect for detect in self.queue)
        tic = time.time()
        if count_true == (self.window-1) and not self.in_motion:
            self.in_motion = True
            state_pub_dbg({'motion': True})
        elif count_false == (self.window-1) and self.in_motion:
            self.in_motion = False
            state_pub_dbg({'motion': False})
        if self.report:
            dt = 1000*(tic - toc)
            report_msg = "MoA: {} ms"
            debug_log(report_msg.format(round(dt)))


def process_command_line():
    description = "Implements an interface to the Pi Camera over ZeroMQ"
    parser = argparse.ArgumentParser(description=description)
    # Max resolution = [2592, 1944]
    parser.add_argument('-W', "--width", type=int, default=1296,
            help="Image output width resolution spec in pixels")
    parser.add_argument('-H', "--height", type=int, default=972,
            help="Image output height resolution spec in pixels")
    parser.add_argument('-I', "--source-width", type=int, default=0,
            help="Sensor capture width resolution spec in pixels")
    parser.add_argument('-E', "--source-height", type=int, default=0,
            help="Sensor capture height resolution spec in pixels")
    parser.add_argument('-r', "--framerate", type=int, default=15,
            help="Frames per second")
    parser.add_argument('-z', "--zeromq-port", type=int, default=50000,
            help="Base port over which ZeroMQ-based services are listening")
    parser.add_argument('-t', "--report-period", type=float, default=0.5,
            help="Camera configuration messages generation period, in seconds")
    parser.add_argument('-d', "--pictures-dir", default='/dev/shm',
            help="Where the captured images will be stored")
    parser.add_argument('-f', "--flip-horizontal", action="store_true",
            help="Whether to flip the camera horizontally")
    parser.add_argument('-v', "--flip-vertical", action="store_true",
            help="Whether to flip the camera vertically")
    parser.add_argument('-m', "--motion-detect", action="store_true",
            help="Whether to perform motion detection")
    parser.add_argument('-a', "--traffic-report", action="store_true",
            help="Whether to log network traffic statistics")
    parser.add_argument('-o', "--motion-report", action="store_true",
            help="Whether to log motion analysis statistics")
    opt = parser.parse_args()
    opt.source_height = opt.source_height or opt.height
    opt.source_width = opt.source_width or opt.width

    return opt

def signal_handler(signum, frame):
    print("\nShutting down...")
    zmq_ctx.destroy()
    sys.exit()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
