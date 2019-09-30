#!/usr/bin/env python3

import sys
import os.path as pth
import time

import cv2
import numpy as np
import zmq

import tchebyshev_img as tc


def main():

    ctx = zmq.Context()
    subs = ctx.socket(zmq.SUB)
    subs.setsockopt(zmq.SUBSCRIBE, b'')
    subs.connect(sys.argv[1])

    cols, rows = 2448, 2048

    # Keyboard-toggle flags for enabling functionality
    zoom_center = False
    show_focus = True
    show_max = True

    focus_roi = 256
    focus_base_path = pth.expanduser("~/code/focus-detect")
    if focus_roi == -1:
        N, M = rows, cols
    else:
        N, M = [focus_roi]*2
    moment_generator = tc.ImageMomentGenerator(N, M, focus_base_path)

    last = time.time()
    while(True):

        received = subs.recv()

        now = time.time()
        delta_t = now - last
        last = now

        frame = np.fromstring(received, np.uint8)
        frame.resize((rows, cols, 3))
        center = (np.array(frame.shape[:2])/2).astype(int)
        offset = center/2

        if zoom_center:
            to_display = frame[center[0]-offset[0]:center[0]+offset[0],
                               center[1]-offset[1]:center[1]+offset[1], :]
        else:
            to_display = cv2.resize(frame, (int(cols/2), int(rows/2)))

        text_line = 0
        add_text(to_display, "Freq [Hz]: {:.1f}".format(1/delta_t),
                 (20, 30+text_line*40))
        text_line += 1

        if show_max:
            text_line = add_stat_info(frame, to_display, text_line)
        if show_focus:
            text_line = add_image_focus(frame, to_display, text_line,
                                        moment_generator, focus_roi)

        cv2.imshow('frame', to_display)
        char_code = cv2.waitKey(2) & 0xFF

        # Exit on the 'q' or Esc keys
        if char_code in (ord('q'), 27):
            break
        # Zoom on center on the 'z' key
        elif char_code == ord('z'):
            zoom_center = not zoom_center
        # Show value statistics on the 's' key
        elif char_code == ord('s'):
            show_max = not show_max 
        # Show focus measure on the 'f' key
        elif char_code == ord('f'):
            show_focus = not show_focus
        elif char_code < 255:
            print("Pressed: {:d}".format(char_code))

    return 0


def add_text(image, text, position, color=[180]*3):

    font = cv2.FONT_HERSHEY_SIMPLEX
    image[max(0, position[1]-27):position[1]+4, position[0]:(position[0]+19*len(text)), :] = 0
    cv2.putText(image, text, tuple(position), font, 1, tuple(color), 2, cv2.LINE_AA)


def add_stat_info(image, canvas, text_line):

    value_chan = image.max(axis=2)
    ch_cdf_norm = get_cdf(value_chan)

    percentile = 50
    ch_median = np.abs(ch_cdf_norm - percentile*1e-2).argmin()

    add_text(canvas, "Med Val: {}".format(ch_median), (20, 30+text_line*40))
    text_line += 1

    percentile = 99
    ch_upper = np.abs(ch_cdf_norm - percentile*1e-2).argmin()

    add_text(canvas, "Max Val: {}".format(ch_upper), (20, 30+text_line*40))
    text_line += 1

    return text_line


def add_image_focus(image, canvas, text_line, moment_generator, center_roi):

    img = ( image if center_roi == -1 else
           crop_center_chunk(image, center_roi) )
    img_norm = tc.normalize_rgb_image(img)
    moments = moment_generator.generate_moments(img_norm, 10)
    focus_measure = tc.focus_measure(moments)

    add_text(canvas, "Focus: {:.4g}".format(focus_measure), (20, 30+text_line*40))
    text_line += 1

    return text_line 


def crop_center_chunk(image, chunk_length=1024):

    center = np.array(image.shape[:2])/2
    center_crop_delta = np.r_[chunk_length, chunk_length]/2

    corner_start = (center - center_crop_delta).astype(int)
    corner_end = (center + center_crop_delta).astype(int)

    return crop_to_bounding_box(image, [corner_start, corner_end])


def crop_to_bounding_box(image, bounding_box):

    min_x, max_x = sorted(co[0] for co in bounding_box)
    min_y, max_y = sorted(co[1] for co in bounding_box)

    im_roi = image[min_y:max_y, min_x:max_x]

    return im_roi


def get_cdf(channel):

    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf/cdf.max()

    return cdf_norm


if __name__ == "__main__":
    sys.exit(main())
