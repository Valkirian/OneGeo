# -*- coding: utf-8 -*-
import json
import gevent
import numpy as np
import zmq.green as zmq
import os
import cv2
from scipy import interpolate

import common

debug_log = common.DebugLog()

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


"""
==================
    cv-tools
==================
"""

def stream_to_cv2(data):
    data_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)

    return image


def cv2_to_stream(image, kind="jpg", quality=95):
    retval, buff = cv2.imencode('.' + kind, image,[cv2.IMWRITE_JPEG_QUALITY, quality,cv2.IMWRITE_PNG_COMPRESSION, 2])
    return buff.tostring()


def file_to_cv2(source_file):
    real_file = os.path.realpath(source_file)
    if not os.path.exists(real_file):
        raise IOError("File {} does not exist".format(source_file))

    return cv2.imread(real_file, cv2.IMREAD_COLOR)


def cv2_to_file(image, target_file, quality=95):
    return cv2.imwrite(target_file, image, [cv2.IMWRITE_JPEG_QUALITY, quality,cv2.IMWRITE_PNG_COMPRESSION, 2])

def color_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_resize(image, percentage=100):
    if percentage != 100:
        size = np.array(image.shape[:2])
        new_size = (percentage*size/100.0).astype(int)
        resized = cv2.resize(image, tuple(new_size[::-1]))
    else:
        resized = image

    return resized


def image_load_resize(image_file, percentage=100):
    image = file_to_cv2(image_file)
    return image_resize(image, percentage)


def precondition_blankfield(bkg_in, max_level=255):
    offset = max_level - np.max(bkg_in)
    bkg_out = bkg_in + offset

    return bkg_out


def blankfield_linear_correct(img_in, bkg_in):
    inv_bkg_in = np.abs(np.max(bkg_in) - bkg_in)
    img_out = cv2.add(inv_bkg_in, img_in)

    return img_out


def blankfield_divide_correct(img_in, bkg_in):
    im_i = cv2.divide(img_in.astype(np.float), bkg_in.astype(np.float))
    return np.uint8((255/im_i.max()) * im_i)


def simple_blur(image, alpha=0.3):
    h, w = image.shape[:2]
    l = 2*int(0.5 * alpha * min(h, w)) + 1
    blurred = cv2.GaussianBlur(image, (l, l), 0)

    return blurred


def blankfield_dark_correct(img_in, bkg_in, pre_blurred=False):
    bfield_blur = bkg_in if not pre_blurred else simple_blur(bkg_in)
    bf_chans = cv2.split(bfield_blur)
    bf_mins = [np.abs(get_cdf(chan) - 1e-2).argmin() for chan in bf_chans]
    bf_chans_deltas = [ch - ch_min for ch, ch_min in zip(bf_chans, bf_mins)]
    bfield_delta = cv2.merge(bf_chans_deltas)
    """
    bf_min = min(bf_mins)
    bf_chans_deltas = [ch - bf_min for ch in bf_chans]
    """
    #c_img = img_in.astype(int) - bfield_blur
    c_img = img_in.astype(int) - bfield_delta
    c_img[c_img < 0] = 0

    return c_img.astype('uint8')


def blankfield_generate_flat(bkg_in):
    h, s, v = cv2.split(cv2.cvtColor(bkg_in, cv2.COLOR_BGR2HSV))
    h_new = (np.median(h) * np.ones_like(h)).astype('uint8')
    s_new = (np.median(s) * np.ones_like(s)).astype('uint8')
    v_new = (np.abs(get_cdf(v) - 5e-2).argmin()*np.ones_like(v)).astype('uint8')

    print("hnew: {}, snew: {}, vnew: {}".format(h_new.min(), s_new.min(),v_new.min()))
    bfi_new = cv2.cvtColor(cv2.merge((h_new, s_new, v_new)), cv2.COLOR_HSV2BGR)
    rel_corr = bfi_new.astype(float) / bkg_in

    return rel_corr


def blankfield_weighted_correct(img_in, wgt_in):
    img_c = np.round(wgt_in * img_in)
    if np.any(img_c > 255):
        print("Image overflowed when correcting by weight")
        img_c[img_c > 255] = 255

    return img_c.astype('uint8')


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def simple_grayscale_stretch(image):
    avg = image.astype(float)
    stretched = 255*(avg - avg.min())/(avg.max() - avg.min())

    return stretched.astype(np.uint8)


def stretch_value_channel(image, lower_pct=5, upper_pct=99):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v_corr = stretch_channel(v, lower_pct, upper_pct)
    img_equ_hsv = cv2.merge([h, s, v_corr.astype('uint8')])
    img_equ = cv2.cvtColor(img_equ_hsv, cv2.COLOR_HSV2BGR)

    return img_equ


def stretch_channel(channel, lower_pct=5, upper_pct=99):
    v = channel
    ch_cdf_norm = get_cdf(v)
    ch_lower = np.abs(ch_cdf_norm - lower_pct*1e-2).argmin()
    ch_upper = np.abs(ch_cdf_norm - upper_pct*1e-2).argmin()
    ch_corr_stretch = 0 + ((v.astype(float) - ch_lower)*(v.max()-v.min()))/(ch_upper-ch_lower)
    ch_corr = ch_corr_stretch.round().astype(int)
    ch_corr[ch_corr < 0] = 0
    ch_corr[ch_corr > 255] = 255

    return ch_corr.astype('uint8')


def get_cdf(channel):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf/cdf.max()

    return cdf_norm


def is_image_dark(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v_cdf_norm = get_cdf(v)
    v_lower = np.abs(v_cdf_norm - 1e-2).argmin()

    return v_lower < 127


# Compute the entropy of a grayscale image
def simple_entropy(image, numbins=100):
    img_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return entropy_single_channel(img_g, numbins)


def entropy_single_channel(channel, numbins):
    # Image is x, only channel 0, no mask, 100 bins, range of [0,256)
    freq = cv2.calcHist([channel], [0], None, [numbins], [0,256])
    pdf = freq/freq.sum()
    pdf_nz = pdf[pdf != 0]
    H_diff = pdf_nz * np.log2(pdf_nz)
    H = -H_diff.sum()

    return H


def entropy(image, subdivisions=10):
    h, w = image.shape[:2]
    stride_r = h/subdivisions
    stride_c = w/subdivisions
    subregions = [image[stride_r*i:min(stride_r*(i+1), h),stride_c*j:min(stride_c*(j+1), w), :]
                  for i in range(subdivisions)
                  for j in range(subdivisions)]
    entropies = [simple_entropy(subregion, 256) for subregion in subregions]

    return np.array(entropies)


def show_image_wait(img):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_spline_lut(interp_points):
    x, y = zip(*interp_points)
    spline = interpolate.interp1d(x, y, kind='cubic')
    lut = spline(range(256))
    lut[lut < 0] = 0
    lut[lut > 255] = 255

    return lut.astype(np.uint8)


def image_apply_lut(image_bgr, lut):
    h, s, v = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV))
    v_corr = cv2.LUT(v, lut)
    im_corr = cv2.cvtColor(cv2.merge((h, s, v_corr)), cv2.COLOR_HSV2BGR)

    return im_corr
