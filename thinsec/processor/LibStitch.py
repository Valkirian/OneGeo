#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import shutil

from common import (img_data_fmt, ensure_dir)


def stitch_as_row(exec_fun, dimensions, stitch_dir, out_dir):
    """
    exec_fun can be any of: os.system, subprocess.call, print, etc...
    """

    hfov = 15
    out_file = os.path.join(out_dir, "row." + img_data_fmt)
    os.chdir(stitch_dir)

    commands = ["pto_gen -p 0 --fov={} -o project.pto *.jpg".format(hfov),
                "cpfind -o project.pto --multirow project.pto",
                "pto_var --opt y -o project.pto project.pto",
                "autooptimiser -v {} -n -o project.pto project.pto".format(hfov),
                "pano_modify -c -s --fov=AUTO --canvas=AUTO -o project.pto project.pto",
                "cat project.pto",
                ]

    for command in commands:
        exec_fun(command)

    roi = gen_roi(get_image_geometry("project.pto"))

    commands = ["nona -v -z DEFLATE -r ldr -m PNG_m -o project project.pto",
                "enblend -f{} --compression=deflate -o pretrim.png -- project*.png".format(roi),
                "convert -trim pretrim.png {}".format(out_file)
                ]

    for command in commands:
        exec_fun(command)

    size = 0 if not os.path.exists(out_file) else os.stat(out_file).st_size
    success = (size > 0)

    return out_file, success


def stitch_as_column(exec_fun, dimensions, stitch_dir, out_dir):

    out_file = os.path.join(out_dir, "image." + img_data_fmt)
    os.chdir(stitch_dir)

    hfov = 10

    commands = ["pto_gen -p 0 --fov={} -o project.pto *.{}".format(hfov, img_data_fmt),
                "cpfind -o project.pto --multirow project.pto",
                "pto_var --opt y,TrX,TrY -o project.pto project.pto",
                "autooptimiser -v {} -n -o project.pto project.pto".format(hfov),
                "pano_modify -p 0 -c -s --fov=AUTO --crop=AUTO --canvas=AUTO -o project.pto project.pto",
                "cat project.pto",
                ]

    for command in commands:
        exec_fun(command)

    roi = gen_roi(get_image_geometry("project.pto"))

    commands = ["nona -v -z DEFLATE -r ldr -m PNG_m -o project project.pto",
                "enblend -f{} --compression=deflate -o pretrim.png -- project*.png".format(roi),
                "convert -trim pretrim.png {}".format(out_file)
                ]

    for command in commands:
        exec_fun(command)

    size = 0 if not os.path.exists(out_file) else os.stat(out_file).st_size
    success = (size > 0)

    return out_file, success


def copy_files(fnames, src_dir, dest_dir):

    ensure_dir(dest_dir)
    for name in fnames:
        shutil.copy(os.path.join(src_dir, name), dest_dir)


def get_image_geometry(project_file):

    with open(project_file) as fobj:
        for line in fobj:
            if line.startswith("p "):
                fields = line.split(' ')
                interest = { f[0]: [int(a) for a in f[1:].split(',')] for f in
                             fields if f.startswith(('w', 'h', 'S')) }
                return interest
    return None


def gen_roi(geometry):

    if 'S' not in geometry:
        return "{}x{}".format(geometry['w'][0], geometry['h'][0])
    else:
        x = geometry['S'][0]
        w = geometry['S'][1] - x
        y = geometry['S'][2]
        h = geometry['S'][3] - y
        return "{}x{}+{}+{}".format(w, h, x, y)
