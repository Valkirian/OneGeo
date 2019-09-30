#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
from glob import glob
import json
import multiprocessing as mp
import os.path as pth
import sys

import cv2
import numpy as np

from common import (img_data_fmt, parse_image_grid_list)

def main():

    opt = process_command_line()
    print opt

    kinds = {'xpl', 'ppl'}
    cell_dirs_kind = { kind: sorted(glob(pth.join(opt.stage_dir, kind, "*"))) for kind in kinds }
    sample_stats = { kind: {pth.basename(ang): dict()} for kind, angles
                    in cell_dirs_kind.items() for ang in angles }

    for kind, cell_dirs in cell_dirs_kind.items():
        for cell_dir in cell_dirs:
            files = glob(pth.join(cell_dir, "*_*." + img_data_fmt))
            n_rows, n_cols, row_cells, img_type, rows, _, _ = parse_image_grid_list(files)

            cell_job = {}
            pool = mp.Pool(processes=opt.threads)
            for i, row in enumerate(row_cells):
                for j, cell in enumerate(row):
                    cell_job[(i, j)] = pool.apply_async(get_stat_info, (cell,))
            pool.close()
            pool.join()

            cell_ang = pth.basename(cell_dir)
            sample_stats[kind][cell_ang] = { str(coord): job.get() for coord, job in cell_job.items() }

    with open(opt.output_file, 'w') as fobj:
        json.dump(sample_stats, fobj)



def get_stat_info(image_file):

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    value_chan = image.max(axis=2)
    ch_cdf_norm = get_cdf(value_chan)
    percentile = lambda p: np.abs(ch_cdf_norm - p*1e-2).argmin()
    ch_median = percentile(50)
    ch_upper = percentile(99)

    return ch_median, ch_upper


def get_cdf(channel):

    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf/cdf.max()

    return cdf_norm


def parse_sample_stats(json_file):

    with open(json_file) as fobj:
        sample_stats_raw = json.load(fobj)

    parse_coord = lambda c: tuple(int(x) for x in c.replace('(', '').replace(')', '').replace(' ', '').split(',') )
    sample_stats = { kind: { ang : { parse_coord(coord): data for coord, data in stats.items() }
                            for ang, stats in angles.items() }
                    for kind, angles in sample_stats_raw.items() }

    return sample_stats


def process_command_line():

    description = "Analyze a null-sample photographic sweep via command line"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("stage_dir",
            help="Where the cell images are")
    parser.add_argument("output_file",
            help="JSON Output file path")
    parser.add_argument('-t', "--threads", type=int,
                        default=mp.cpu_count(),
            help=("Maximum number of simultaneous processes to execute"))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    sys.exit(main())
