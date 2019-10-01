#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import distutils.dir_util as dir_util
from glob import glob
from multiprocessing import (cpu_count, Pool)
import os
import os.path as pth
import re
import shutil
import tempfile

import cv2
import numpy as np
import pyinotify

import black_borders_detect as bdd
from common import (DebugLog, img_data_fmt, get_digits,
                    parse_image_grid_list)
from cv_tools import (file_to_cv2, image_resize, cv2_to_file,
                      blankfield_linear_correct)
from ioevent import FilesWaiter
from LibStitch import ensure_dir
from sweep_processor import (command_executor_factory, setup_directories)

debug_log = DebugLog()


"""
An example for manually invoking Fiji:

fiji --headless -eval 'run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x=23 grid_size_y=20 tile_overlap=25 first_file_index_x=0 first_file_index_y=0 directory=/home/worker/4x_trans/cells_40pc/ file_names={yy}_{xx}.jpg output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]"); saveAs("Jpeg", "/home/worker/4x_trans/cells_40pc/4x_40pc.jpg"); close(); "Exit";'
"""

fiji_save_formats = {'jpg': "Jpeg", 'png': "PNG"}

fiji_command_template = "fiji --allow-multiple --headless -eval '{}'"
# Used for stitching as a grid defined by filename
fiji_grid_stitch_inst_tpl = ('run("Grid/Collection stitching", "{}"); '
                             'eval("script", "System.exit(0);");')
fiji_grid_compute_arg_tpl = ('type=[Filename defined position] '
                             'order=[Defined by filename         ] '
                             'grid_size_x={n_cols:d} grid_size_y={n_rows:d} '
                             'tile_overlap={tile_overlap} '
                             'first_file_index_x={first_col:d} '
                             'first_file_index_y={first_row:d} '
                             'directory={input_dir_path} '
                             'file_names={{{row_spec}}}_{{{col_spec}}}.{img_ext} '
                             'output_textfile_name={out_txt_file} '
                             'fusion_method=[{fusion_method}] '
                             'regression_threshold={threshold_regression:.2f} '
                             'max/avg_displacement_threshold={threshold_maxavg:.2f} '
                             'absolute_displacement_threshold={threshold_abs:.2f} '
                             'compute_overlap '
                             'computation_parameters=[Save computation time (but use more RAM)] ')
fiji_fusion_options = {'none': "Do not fuse images (only write TileConfiguration)",
                       'linear': "Linear Blending"}
fiji_fuse_arg_tpl = ('type=[Positions from file] '
                     'order=[Defined by TileConfiguration] '
                     'directory={input_dir_path} '
                     'layout_file={tile_file} ' # TileConfiguration.registered.txt
                     'fusion_method=[Linear Blending] '
                     'regression_threshold={threshold_regression:.2f} '
                     'max/avg_displacement_threshold={threshold_maxavg:.2f} '
                     'absolute_displacement_threshold={threshold_abs:.2f} '
                     'computation_parameters=[Save memory (but be slower)] ')
fiji_write_arg_tpl = 'image_output=[Write to disk] output_directory={} '


def fiji_grid_stitch(directories, files, out_file, do_fuse, do_crop=False,
                     threshold_reg=0.3, threshold_maxavg=2, threshold_abs=3,
                     tile_overlap=25, keep_uncropped=False, pre_copy_files=True,
                     print_log_to_stdout=False):

    out_file_name = os.path.splitext(out_file)[0]
    out_file_path = "{}.{}".format(out_file_name, img_data_fmt)
    pre_crop_fname = "{}.precrop.{}".format(os.path.basename(out_file_name),
                                            img_data_fmt)
    pre_crop_file = os.path.join(directories['out'], pre_crop_fname)
    base_dir = os.path.dirname(out_file_path)

    fusion_method = fiji_fusion_options['linear' if do_fuse else 'none']

    file_list = files
    if pre_copy_files:
        debug_log("Copying source images into", directories['in'])
        file_list = []
        for _file in files:
            shutil.copy(_file, directories['in'])
            file_list.append(os.path.join(directories['in'],
                                          os.path.basename(_file)))
    input_path = set(os.path.dirname(fi) for fi in file_list).pop()

    (n_rows, n_cols, row_cells, img_type, _,
                     digits_row, digits_col) = parse_image_grid_list(file_list)

    debug_log("Assembling grid", "into", out_file_path, "on",
              directories['in'], "started")

    first_cell = os.path.basename(row_cells[0][0])[:-4]
    first_row, first_col = map(int, first_cell.split('_'))

    out_txt_filename = os.path.basename(out_file_name)
    gen_txt_file = (os.path.join(input_path, out_txt_filename)
                     + ".registered.txt")

    fiji_args = fiji_grid_compute_arg_tpl.format(n_cols=n_cols, first_col=first_col,
                                                 n_rows=n_rows, first_row=first_row,
                                                 row_spec='y'*digits_row,
                                                 col_spec='x'*digits_col,
                                                 input_dir_path=input_path,
                                                 img_ext=img_type,
                                                 fusion_method=fusion_method,
                                                 out_txt_file=out_txt_filename,
                                                 tile_overlap=tile_overlap,
                                                 threshold_regression=threshold_reg,
                                                 threshold_maxavg=threshold_maxavg,
                                                 threshold_abs=threshold_abs)
    fiji_write = ("" if not do_fuse else
                  fiji_write_arg_tpl.format(directories['out']))
    fiji_inst = fiji_grid_stitch_inst_tpl.format(fiji_args + fiji_write)
    command = fiji_command_template.format(fiji_inst)

    print("Command is:")
    print(command)

    exec_fun = command_executor_factory(directories['log'], pre_crop_fname,
                                        print_log_to_stdout)
    retcode = exec_fun(command)
    success = (not bool(retcode))

    log_dir = ensure_dir(os.path.join(base_dir, 'log'))
    dir_util.copy_tree(directories['log'], log_dir)

    result_name = "done" if success else "failed"
    debug_log("Assembling of", pre_crop_file, result_name, retcode)

    out_txt_file, matrix, image = [None]*3

    if success:

        out_txt_file = gen_txt_file

        if do_fuse:
            channel_files = sorted(glob(os.path.join(directories['out'],
                                                     "img_t1_z1_c?")))
            channels = [ cv2.imread(chan, cv2.IMREAD_GRAYSCALE) for chan
                        in channel_files ]
            image_precrop = cv2.merge(channels[::-1])

            if do_crop:
                debug_log("Straigthening and Cropping", pre_crop_file, "into",
                        out_file_path)
                matrix, image = straighten_crop(image_precrop, True)
            else:
                image = image_precrop

            success = cv2_to_file(image, out_file_path)
            result_name = "done" if success else "failed"
            debug_log("Assembly of", out_file_path, result_name)

        if not keep_uncropped:
            work_dir = os.path.dirname(directories['in'])
            debug_log("Removing work directory", work_dir)
            shutil.rmtree(work_dir)

    return success, out_txt_file, matrix, image, None


def fiji_grid_fuse(directories, tiles_file, out_file, threshold_reg=0.3,
                   threshold_maxavg=2, threshold_abs=3, do_crop=False):

    source_dir, tiles_full_fname = os.path.split(tiles_file)

    out_file_name = os.path.splitext(out_file)[0]
    out_file_path = "{}.{}".format(out_file_name, img_data_fmt)
    pre_crop_fname = "{}.precrop.{}".format(os.path.basename(out_file_name),
                                            img_data_fmt)
    pre_crop_file = os.path.join(directories['out'], pre_crop_fname)
    base_dir = os.path.dirname(out_file_path)

    fiji_args = fiji_fuse_arg_tpl.format(input_dir_path=source_dir,
                                         tile_file=tiles_full_fname,
                                         threshold_regression=threshold_reg,
                                         threshold_maxavg=threshold_maxavg,
                                         threshold_abs=threshold_abs)
    fiji_write = fiji_write_arg_tpl.format(directories['out'])
    fiji_inst = fiji_grid_stitch_inst_tpl.format(fiji_args + fiji_write)
    command = fiji_command_template.format(fiji_inst)

    # Establish inotify watches for ensuring output channels are fully written
    # before proceding with merging
    channel_files = [ pth.join(directories['out'], "img_t1_z1_c{}".format(i))
                     for i in range(1, 4) ]
    wm = pyinotify.WatchManager()
    watches = FilesWaiter(channel_files, wm)

    # Run fusion
    print("Command is:")
    print(command)
    exec_fun = command_executor_factory(directories['log'], pre_crop_fname)
    retcode = exec_fun(command)
    success = (not bool(retcode))
    result_name = "done" if success else "failed"

    log_dir = os.path.join(base_dir, 'log')
    dir_util.copy_tree(directories['log'], log_dir)

    debug_log("Assembling of", pre_crop_file, result_name, retcode)
    matrix, image = [None]*2
    channels_done = watches.wait()
    if success and channels_done:

        channels = [cv2.imread(chan, cv2.IMREAD_GRAYSCALE) for chan in channel_files]
        [os.remove(chan) for chan in channel_files]
        image_precrop = cv2.merge(channels[::-1])

        if do_crop:
            debug_log("Straigthening and Cropping", pre_crop_file, "into",
                    out_file_path)
            matrix, image = straighten_crop(image_precrop, True)
        else:
            image = image_precrop

        success = cv2_to_file(image, out_file_path)
        result_name = "done" if success else "failed"
        debug_log("Assembly of", out_file_path, result_name)

    return success, image, matrix


class TileConfigurator(object):

    def __init__(self, dimensions=2):

        self.dimensions = dimensions

    def parse(self, file_path):

        displacements = {}
        disp_re = re.compile(r'^([^.]+\.[^;]+); ; \(([0-9E.-]+), ([0-9E.-]+)\)')
        with open(file_path) as fobj:
            for line in fobj:
                search = disp_re.search(line.strip())
                if search is not None:
                    filename, dx, dy = search.groups()
                    displacements[filename] = np.r_[float(dx), float(dy)]
                #TODO add parsing of dimensions

        return displacements

    def generate(self, displacements, filename):

        lines = []

        lines.append("dim = {}".format(self.dimensions))
        lines.append('')

        lines.extend("{}; ; ({:.12f}, {:.12f})".format(fname, dx, dy)
                     for fname, (dx, dy) in sorted(displacements.items()))

        with open(filename, 'w') as fobj:
            lines.append('')
            fobj.write('\n'.join(lines))
        debug_log("Tile file", filename, "generated")


def assemble_multigrid(directories, files, out_file_name, pre_resize=100,
                       base_cell_size=4, max_threads=cpu_count(),
                       keep_uncropped=False, blankfield_file=None):

    n_rows, n_cols, row_cells, img_type, _, _, _ = parse_image_grid_list(files)
    pool = Pool(processes=max_threads)

    cs = float(base_cell_size)
    base_cells_dims = np.ceil(np.r_[n_rows, n_cols]/cs).astype(int)

    cells_dir = ensure_dir(os.path.join(directories['out'], "cells"))
    cell_jobs = []
    cell_files = []

    blankfield = (None if blankfield_file is None else
                  image_resize(file_to_cv2(blankfield_file), pre_resize))

    digits_row, digits_col = map(get_digits, base_cells_dims)

    for b_row in range(base_cells_dims[0]):
        for b_col in range(base_cells_dims[1]):

            base_name = "{:0{w_r}d}_{:0{w_c}d}".format(b_row, b_col,
                                                       w_r=digits_row,
                                                       w_c=digits_col)
            temp_dirs = setup_directories(os.path.join(cells_dir, base_name))
            files = [file_image_preprocess(row_cells[i][j],
                                           os.path.join(temp_dirs['in'],
                                                        os.path.basename(row_cells[i][j])),
                                           pre_resize,
                                           blankfield
                                       )
                     for i in range(base_cell_size*b_row,
                                    min(base_cell_size*(b_row+1), n_rows))
                     for j in range(base_cell_size*b_col,
                                    min(base_cell_size*(b_col+1), n_cols)) ]
            out_file = os.path.join(cells_dir,
                                    "{}.{}".format(base_name, img_data_fmt))
            cell_jobs.append(pool.apply_async(fiji_grid_stitch,
                                              (temp_dirs, files, out_file),
                                              {'pre_copy_files': False,
                                               'keep_uncropped': keep_uncropped}))
            cell_files.append(out_file)
    pool.close()
    pool.join()

    cell_successes = [job.get() for job in cell_jobs]
    if all(cell_successes):
        debug_log("Assembling of base cells into", cells_dir, "is complete.",
                  "Assembling image...")
        return fiji_grid_stitch(directories, cell_files, out_file_name,
                                keep_uncropped=keep_uncropped)
    else:
        return False


def file_image_preprocess(source_file, target_file, reduction_percentage=100,
                          blankfield=None):

    if reduction_percentage == 100 and blankfield is None:
        shutil.copyfile(source_file, target_file)
    else:
        image = image_resize(file_to_cv2(source_file), reduction_percentage)
        corrected = (image if blankfield is None else
                     blankfield_linear_correct(image, blankfield))
        cv2_to_file(corrected, target_file, 99)

    return target_file


def straighten_crop(image, assume_well_aligned=True, *args, **kwargs):

    if not assume_well_aligned:
        return straighten_crop__original(image, *args, **kwargs)

    h, w = image.shape[:2]

    #im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #band_h, band_v = [ int(0.1*x) for x in (h, w) ]
    #left = bdd.find_left_border(im, band_v)
    #top = bdd.find_top_border(im, band_h)
    #right = bdd.find_right_border(im, band_v)
    #bottom = bdd.find_bottom_border(im, band_h)
    #out_im = bdd.crop_borders(image, (left, right, top, bottom))

    border_s = 50
    border_spec = [border_s, w - border_s, border_s, h - border_s]
    out_im = bdd.crop_borders(image, border_spec)
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    return M, out_im


def straighten_crop__original(image, pcx=0.006, pcy=0.004):

    h, w, z = image.shape
    # Extra pixels
    l = 200
    im2 = np.zeros([h + l, w + l, z]).astype(image.dtype)
    im2[l/2:h+l/2, l/2:w+l/2] = image[:,:]
    image = im2

    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im, 5, 255,0)

    thresh = cv2.bilateralFilter(thresh, 9, 75, 75)
    kernel = np.ones((100, 100), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((150, 150), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations = 1)
    thresh = cv2.erode(thresh, kernel, iterations = 1)

    ret,thresh = cv2.threshold(thresh, 250, 255,0)
    # edged = cv2.Canny(thresh, 30, 100)
    # show_image_wait(thresh)

    im2, contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    cnt = cnts[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Sort order: top-left, bottom-left, top-right, bottom-right
    box_s = np.append(  box[box[:2, 1].argsort()],
                        box[box[2:, 1].argsort() + 2], axis=0 )

    box_h = np.sqrt(np.dot(box_s[0]-box_s[1], box_s[0]-box_s[1]))
    box_w = np.sqrt(np.dot(box_s[0]-box_s[2], box_s[0]-box_s[2]))

    """
    cv2.drawContours(image, [box], -1, (0, 255, 0), 5)
    show_image_wait(image)
    """

    pts1 = np.float32(box_s)
    pts2 = np.float32([[0, 0], [0, box_h], [box_w, 0], [box_w, box_h]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    out_im = cv2.warpPerspective(image, M,
                            (np.int(np.round(box_w)),
                            np.int(np.round(box_h))))
    h, w, z = out_im.shape
    crop_left = np.int(pcx*w)
    crop_right = np.int(pcx*w)
    crop_top = np.int(pcy*h)
    crop_bottom = np.int(3*pcy*h)
    out_im = out_im[crop_top:-crop_bottom, crop_left:-crop_right]

    return M, out_im


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs='*',
            help="Input image files to be stitched")
    parser.add_argument('-d', "--work-dir", default='/dev/shm',
            help="Where input, temporary, and output files will be written")
    parser.add_argument('-o', "--output-file", default='out.' + img_data_fmt,
            help="Pathname of output file")
    parser.add_argument('-g', "--multi-grid", action='store_true',
            help=("Perform stitching in two grid steps (multiple grids) or "
                  "a single one"))
    parser.add_argument('-p', "--pre-resize", default=100, type=int,
            help=("When stitching via multi-grid, the percentage to resize each "
                  "input image to"))
    parser.add_argument('-b', "--blankfield-file", default=None,
            help=("When stitching via multi-grid, the image of the blank field "
                  "which will be used to correct each input image's lighting"))
    parser.add_argument('-s', "--mg-cell-size", default=4, type=int,
            help=("When stitching via multi-grid, the amount of rows and columns"
                  " to fuse into each super cell for the second grid"))
    parser.add_argument('-t', "--threads", type=int,
                        default=max(1, cpu_count() - 1),
            help=("Maximum number of simultaneous processes to execute"))
    parser.add_argument('-r', "--remove-work-tree", action='store_true',
            help=("Remove intermediate files if the stitch is successful"))
    parser.add_argument('-k', "--keep-uncropped", action='store_true',
            help=("Keep uncropped intermediate images"))

    opt = parser.parse_args()
    print opt
    #TODO receive arguments for Fiji's stitching thresholds

    work_dir = tempfile.mkdtemp(dir=opt.work_dir)
    work_dirs = setup_directories(work_dir)
    debug_log("JOB", "Staging files in", work_dir)
    output_file = os.path.abspath(opt.output_file)

    if opt.multi_grid:
        success = assemble_multigrid(work_dirs, opt.files, output_file,
                                     opt.pre_resize, opt.mg_cell_size,
                                     opt.threads, opt.keep_uncropped,
                                     opt.blankfield_file)
    else:
        success = fiji_grid_stitch(work_dirs, opt.files, output_file,
                                   keep_uncropped=opt.keep_uncropped)

    if success and opt.remove_work_tree:
        shutil.rmtree(work_dir)
