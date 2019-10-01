#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np

from common import DebugLog
import cv_tools as cvt

debug_log = DebugLog()


def main():

    opt = process_command_line()
    print opt

    avg = cvt.file_to_cv2(os.path.expanduser(opt.files[0])).astype(float)
    for img_file in opt.files[1:]:
        avg += cvt.file_to_cv2(os.path.expanduser(img_file))
    avg /= len(opt.files)

    success = cvt.cv2_to_file(avg.astype(np.uint8), opt.output)
    result = "done" if success else "failed"
    debug_log("Average image construction into", opt.output, result)


def process_command_line():

    description = ("Finds the optimal common registration among a set of images"
                   "by computing relative translations")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", nargs='*',
            help="Input image files to be averaged")
    parser.add_argument('-o', "--output",
            help=("Output file"))

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
