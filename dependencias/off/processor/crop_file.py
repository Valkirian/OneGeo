#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv_tools as cvt


def main():

    args = process_command_line()

    in_image = cvt.file_to_cv2(args['file-in'])
    border_h = args['border-horizontal']
    border_v = args['border-vertical']
    out_image = in_image[border_h:-border_h, border_v:-border_v, :]
    cvt.cv2_to_file(out_image, args['file-out'])


def process_command_line():

    description = "Crops an image's borders"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("border-horizontal", type=int,
            help="Pixels to remove on top and bottom borders")
    parser.add_argument("border-vertical", type=int,
            help="Pixels to remove on left and right borders")
    parser.add_argument("file-in",
            help="Input image file")
    parser.add_argument("file-out",
            help="Output image file")

    return parser.parse_args()


if __name__ == "__main__":
    main()
