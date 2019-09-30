#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import distutils.dir_util as dir_util
from glob import glob
from math import ceil
import itertools as it
import multiprocessing as mp
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

from black_borders_detect import (find_left_border, find_top_border)
from block_reduction import (mp_process_blockreduce, BlockReducer)
from common import (DebugLog, img_data_fmt, parse_image_grid_list,
                    ensure_dir, shell_retcode)
from corner_blending import patch_body_corner_inmem
from cv_tools import (file_to_cv2, cv2_to_file, color_to_gray, image_resize,
                      simple_grayscale_stretch)
from fiji_driver import (fiji_grid_stitch, TileConfigurator, fiji_grid_fuse)
from LibStitch import (stitch_as_row, stitch_as_column)
from sweep_processor import (command_executor_factory, setup_directories)

debug_log = DebugLog()

job_types = {'row', 'col', 'image-by-rows', 'image-by-blocks', 'two-step',
             'multi-step', 'block-links'}


def main():

    opt = process_command_line()
    print opt

    work_dir = tempfile.mkdtemp(dir=opt.work_dir)
    work_dirs = setup_directories(work_dir)
    debug_log("JOB", "Staging files in", work_dir)
    output_file = os.path.abspath(opt.output_file)
    img_path, img_fname_ext = os.path.split(output_file)
    img_fname, img_ext = os.path.splitext(img_fname_ext)

    if opt.job_type == 'row':
        success = assemble_row(work_dirs, opt.files,
                               output_file)
    elif opt.job_type == 'col':
        success = assemble_column(work_dirs, opt.files,
                                  output_file)
    elif opt.job_type == 'image-by-rows':
        success = assemble_image_by_rows(work_dirs, opt.files,
                                         output_file, opt.threads)
    elif opt.job_type == 'image-by-blocks':
        success = assemble_image_by_blocks(work_dirs, opt.files,
                                           output_file, opt.threads)
    elif opt.job_type == 'two-step':
        success = assemble_twostep(work_dirs, opt.files, output_file)

    elif opt.job_type == 'multi-step':
        success = assemble_multistep(work_dirs, opt.files, output_file,
                                     keep_files=opt.keep_work_dir,
                                     threads=opt.threads, overlap=opt.tile_overlap)

    elif opt.job_type == 'block-links':
        success, img, _ = assemble_blocklink(work_dirs, opt.files,
                                             output_file, threads=opt.threads,
                                             keep_files=opt.keep_work_dir,
                                             overlap=opt.tile_overlap,
                                             do_contrast_stretch=opt.contrast)

        if success and opt.also_save_reduced < 100:
            img_path, img_fname = os.path.split(output_file)
            reduced_fname = os.path.splitext(img_fname)[0] + ".small.jpg"
            reduced_fpath = os.path.join(img_path, reduced_fname)
            reduced_img = image_resize(img, opt.also_save_reduced)
            cv2_to_file(reduced_img, reduced_fpath)

    if success and not opt.keep_work_dir:
        shutil.rmtree(work_dir)

    return shell_retcode(success)


def assemble_row(directories, files, out_file_name):

    return perform_assembly(directories, files,
                            "row", stitch_as_row, out_file_name)


def assemble_column(directories, files, out_file_name):

    return perform_assembly(directories, files,
                            "column", stitch_as_column, out_file_name)


def assemble_image_by_rows(directories, files, out_file_name, max_threads):

    n_rows, n_cols, row_cells, img_type, rows, _, _ = parse_image_grid_list(files)
    pool = mp.Pool(processes=max_threads)

    rows_dir = ensure_dir(os.path.join(directories['out'], "rows"))
    row_files = [os.path.join(rows_dir, "row_{}.{}".format(row, img_data_fmt))
                 for row in rows]
    temp_dirs = [setup_directories(os.path.join(rows_dir, "row_{}".format(i)))
                 for i in rows]
    row_jobs = [pool.apply_async(assemble_row, (temp_dirs[i], row_cells[i],
                                                row_files[i]))
                for i in range(n_rows)]
    pool.close()
    pool.join()

    row_successes = [job.get() for job in row_jobs]
    if all(row_successes):
        debug_log("Assembling of rows into", rows_dir, "is complete.",
                  "Assembling image...")
        return assemble_column(directories, row_files, out_file_name)
    else:
        return False


def assemble_image_by_blocks(directories, files, out_file_name, max_threads):

    n_rows, n_cols, row_cells, img_type, _, _, _ = parse_image_grid_list(files)

    reducer = BlockReducer(n_rows, n_cols)
    reduction_dir = os.path.join(directories['in'], "reduction")
    reduce_fun = PairStitcher(reduction_dir)

    debug_log("Assembling by blocks of", out_file_name, "in", directories['in'],
              "starting...")
    mp_process_blockreduce(reducer, row_cells, reduce_fun,
                           mp.cpu_count())
    success = reducer.done()

    if success:
        result_file = reducer.result_identifier()
        shutil.copyfile(result_file, out_file_name)
        debug_log("Assembling by blocks of", out_file_name, "is complete.")

    return success


def assemble_twostep(directories, files, out_file_name, corner_dim_min=[2, 3]):

    n_rows, n_cols, row_cells, img_type, _, _, _ = parse_image_grid_list(files)

    body_dir = ensure_dir(os.path.join(directories['out'], "body"))
    debug_log("Assembling of body into", body_dir, "started.")
    body_work_dir = setup_directories(body_dir)
    body_simple_file = os.path.join(body_dir, "body-straight.png")

    (success, tiles_file, body_matrix,
     body_simple, pre_crop_file) = fiji_grid_stitch(body_work_dir, files,
                                                    body_simple_file, True,
                                                    keep_uncropped=True,
                                                    pre_copy_files=False,
                                                    threshold_reg=0.5,
                                                    threshold_maxavg=10,
                                                    threshold_abs=50)
    if not success:
        debug_log("Could not assemble body")
        return False

    c_h, c_w = body_simple.shape[:2]
    grey_chunk = color_to_gray(body_simple[:c_h/2, :c_w/2, :])
    left_black = find_left_border(grey_chunk, c_w/2)
    top_black = find_top_border(grey_chunk, c_h/2)
    pre_corner = body_simple[:top_black, :left_black]
    corner_is_black = (pre_corner.max() == 0)
    debug_log("Corner void size:", left_black, "x", top_black, corner_is_black)

    if (left_black == 0 and top_black == 0) or (not corner_is_black):
        debug_log("Apparently no corner stitching is necessary!")
        shutil.move(body_simple_file, out_file_name)
        return True

    dim_h, dim_w = file_to_cv2(files[0]).shape[:2]
    corner_dim = [max(corner_dim_min[0], 1 + 2*int(ceil(float(top_black)/dim_h))),
                  max(corner_dim_min[1], 1 + 2*int(ceil(float(left_black)/dim_w)))]

    debug_log("Corner dimensions: {}".format(corner_dim))
    corner_dir = ensure_dir(os.path.join(directories['out'], "corner"))
    corner_work_dir = setup_directories(corner_dir)
    corner_file = os.path.join(corner_dir, "corner.png")
    corner_input_files = [ cell for row in row_cells[:corner_dim[0]]
                           for cell in row[:corner_dim[1]] ]
    debug_log("Assembling of corner into", corner_dir, "started.")
    success, _, corner_matrix, corner, _ = fiji_grid_stitch(corner_work_dir,
                                                            corner_input_files,
                                                            corner_file, True,
                                                            pre_copy_files=False)
    if not success:
        debug_log("Could not assemble corner")
        return False

    debug_log("Generating Laplacian blending weights for corner")
    body = patch_body_corner_inmem(body_simple, body_matrix,
                                   corner, corner_matrix)

    debug_log("Saving blended image to", out_file_name)
    success = cv2_to_file(body, out_file_name)
    result_name = "done" if success else "failed"
    debug_log("Saving blended image", out_file_name, result_name)

    return success


def assemble_multistep(directories, files, out_file_name, base_cell_size=3,
                       keep_files=True, threads=1, overlap=25):

    (n_rows, n_cols, row_cells, img_type, _,
                     digits_row, digits_col) = parse_image_grid_list(files)

    # Recursion base case: A single cell remains
    if n_rows == n_cols == 1:
        shutil.move(row_cells[0][0], out_file_name)
        success = True

    # Recursion step
    else:

        blocks = list_grid_block_paths(n_rows, n_cols, row_cells, base_cell_size)

        step_size = len(blocks)
        all_blocks_dir = ensure_dir(os.path.join(directories['temp'], str(step_size)))
        step_done = True
        all_blocks_files = []

        job_args = [ (block_i, block_j, block_rows, img_type, all_blocks_dir,
                      digits_row, digits_col, keep_files, overlap)
                    for (block_i, block_j), block_rows in blocks ]
        if threads == 1:
            results = [ multistep_worker(*args) for args in job_args ]
        else:
            pool = mp.Pool(processes=threads)
            jobs = [pool.apply_async(multistep_worker, args) for args in job_args]
            pool.close()
            pool.join()
            results = [ job.get() for job in jobs ]
        success = all( r['success'] for r in results )
        all_blocks_files = [ r['block-path'] for r in results ]

        if step_done:
            success = assemble_multistep(directories, all_blocks_files, out_file_name,
                                         base_cell_size)

    return success


def multistep_worker(block_i, block_j, block_rows, img_type, blocks_dir,
                     digits_row, digits_col, keep_files, overlap):

    block_coord = "{:0{dig_row}d}_{:0{dig_col}d}".format(block_i, block_j,
                                                         dig_row=digits_row,
                                                         dig_col=digits_col)
    block_file = "{}.{}".format(block_coord, img_type)
    block_dir = setup_directories(os.path.join(blocks_dir, block_coord))
    block_path = os.path.join(blocks_dir, block_file)
    block_cells = sum(block_rows, [])

    success, _, _, _, _ = fiji_grid_stitch(block_dir, block_cells,
                                           block_path, True,
                                           keep_uncropped=False,
                                           pre_copy_files=False,
                                           tile_overlap=overlap)
    if not success:
        debug_log("Could not assemble", block_path)


    if success and not keep_files:
        debug_log("Removing block cells:")
        for cell in block_cells:
            print cell,
            os.remove(cell)
        print
    
    return {'success': success, 'block-path': block_path}


def assemble_blocklink(directories, files, out_file_name, do_fuse=True,
                       base_cell_size=3, threshold_reg=0.3, threshold_maxavg=2,
                       threshold_abs=3, keep_files=True, pre_copy_files=True,
                       do_crop=False, overlap=25, threads=1, do_contrast_stretch=False):
    # Return variables
    success, image, all_displacements = False, None, None

    if not pre_copy_files:
        file_list = files
    else:
        debug_log("Copying source images into", directories['in'])
        file_list = []
        for _file in files:
            shutil.copy(_file, directories['in'])
            file_list.append(os.path.join(directories['in'],
                                          os.path.basename(_file)))

    (n_rows, n_cols, row_cells, img_type, _,
                     digits_row, digits_col) = parse_image_grid_list(file_list)

    blocks = list_grid_block_paths(n_rows, n_cols, row_cells, base_cell_size)
    blocks_cells_d = { bl[0]: bl[1] for bl in blocks }
    links_cells_d = get_link_blocks(blocks_cells_d)

    input_path = set(os.path.dirname(fi) for fi in file_list).pop()

    all_blocks_dir = directories['temp']
    blocks_dim_d = {}
    blocks_disp_d = {}

    cell_dim = cv2.imread(row_cells[0][0], cv2.IMREAD_GRAYSCALE).shape

    # 1. Get intra-block cell displacements
    job_args = [ (block_idx, block_rows, digits_row, digits_col, all_blocks_dir,
                  cell_dim)
                for block_idx, block_rows in blocks ]
    if threads == 1:
        results = [ bl_intrablock_worker(*args) for args in job_args ]
    else:
        pool = mp.Pool(processes=threads)
        jobs = [pool.apply_async(bl_intrablock_worker, args) for args in job_args]
        pool.close()
        pool.join()
        results = [ job.get() for job in jobs ]
    success = all( r['success'] for r in results )
    if not success:
        debug_log("Could not assemble a block")
        return success, image, all_displacements
    blocks_dim_d = {k: v for r in results for k, v in r['dim'].items()}
    blocks_disp_d = {k: v for r in results for k, v in r['disp'].items()}

    def displ_get(block_idx, cell_idx):
        return blocks_disp_d[block_idx][cell_idx[0]][cell_idx[1]]

    # 2. Get linking displacements among blocks' corners
    job_args = [ (block_idx, block_rows, digits_row, digits_col,
                  all_blocks_dir, cell_dim, link_cells_d, overlap)
                for block_idx, link_cells_d in links_cells_d.items() ]
    if threads == 1:
        results = [ bl_links_worker(*args) for args in job_args ]
    else:
        pool = mp.Pool(processes=threads)
        jobs = [pool.apply_async(bl_links_worker, args) for args in job_args]
        pool.close()
        pool.join()
        results = [ job.get() for job in jobs ]
    success = all( r['success'] for r in results )
    if not success:
        debug_log("Could not compute a link")
        return success, image, all_displacements
    links_disp_d = {k: v for r in results for k, v in r['links'].items()}

    # 3. Refer all displacements to the reference frame where block(0,0) lies
    #    at the origin
    displaced_blocks = {(0, 0)}
    for ref_block, link_displacement_d in sorted(links_disp_d.items()):

        # _aco = Absolute coordinate (coordinate in reference frame)
        link_ref_aco = blocks_disp_d[ref_block][-1][-1]

        for tgt_block, link_disp in sorted(link_displacement_d.items()):

            it_is_self = (tgt_block == ref_block)
            it_is_already_displaced = (tgt_block in displaced_blocks)
            if it_is_self or it_is_already_displaced:
                continue

            tgt_cell_aco = link_ref_aco + link_disp

            tgt_cell_linkidx = links_cells_d[ref_block][tgt_block][1]
            tgt_cell_localcoord = displ_get(tgt_block, tgt_cell_linkidx)

            tgt_displacement = tgt_cell_aco - tgt_cell_localcoord
            debug_log("Moving block", tgt_block, "by", tgt_displacement)

            disps_to_update = blocks_disp_d[tgt_block]
            new_disps = [ [ disp + tgt_displacement for disp in row ]
                         for row in disps_to_update ]
            blocks_disp_d[tgt_block] = new_disps
            displaced_blocks.add(tgt_block)

    # 4. Generate global displacements file

    disps_list = [ zip(sum(block_rows, []), sum(blocks_disp_d[block_idx], []))
                  for block_idx, block_rows in blocks ]
    all_displacements = { os.path.basename(cellp): disp for block in disps_list
                         for cellp, disp in block }
    global_tile_file = os.path.join(input_path, "tiles-positions.txt")
    TileConfigurator().generate(all_displacements, global_tile_file)

    if do_fuse:
        # 5. Fuse all cells into the stitch
        success, image, _ = fiji_grid_fuse(directories, global_tile_file,
                                           out_file_name, threshold_reg,
                                           threshold_maxavg, threshold_abs,
                                           do_crop)
        if do_contrast_stretch:
            image_corr = cv2.merge([simple_grayscale_stretch(ch) for ch in cv2.split(image)])
            success = cv2_to_file(image_corr, out_file_name)
            image = image_corr
    else:
        image = None

    if pre_copy_files:
        debug_log("Removing copied source images from", directories['in'])
        shutil.rmtree(directories['in'])

    return success, image, all_displacements


def bl_intrablock_worker(block_idx, block_rows, digits_row, digits_col,
                         all_blocks_dir, cell_dim):

    blocks_dim_d = {}
    blocks_disp_d = {}

    block_coord = "{:0{dig_row}d}_{:0{dig_col}d}".format(*block_idx,
                                                         dig_row=digits_row,
                                                         dig_col=digits_col)
    block_file = "bl-" + block_coord
    block_dir = setup_directories(os.path.join(all_blocks_dir, block_coord))
    block_path = os.path.join(all_blocks_dir, block_file)
    block_cells = sum(block_rows, [])

    success, tiles_file = fiji_grid_stitch(block_dir, block_cells,
                                           block_path, False,
                                           pre_copy_files=False)[:2]
    if not success:
        debug_log("Could not assemble", block_path)
    else:
        displs = TileConfigurator().parse(tiles_file)
        block_stitch_size, block_origind = compute_stitch_size(cell_dim, displs)
        blocks_dim_d[block_idx] = (block_stitch_size, block_origind)
        cell_disp = [ [ displs[os.path.basename(cpath)] - block_origind for cpath
                       in row ] for row in block_rows ]
        blocks_disp_d[block_idx] = cell_disp

    return {'success': success, 'dim': blocks_dim_d, 'disp': blocks_disp_d}


def bl_links_worker(block_idx, block_rows, digits_row, digits_col,
                    all_blocks_dir, cell_dim, link_cells_d, overlap):

    links_disp_d = {}

    lnk_coord = "{:0{dig_row}d}_{:0{dig_col}d}".format(*block_idx, dig_row=digits_row,
                                                       dig_col=digits_col)
    lnk_file = "lk-" + lnk_coord
    lnk_dir = setup_directories(os.path.join(all_blocks_dir, lnk_coord))
    lnk_path = os.path.join(all_blocks_dir, lnk_file)
    lnk_cells = [cell_n for cell_n, cell_idx in link_cells_d.values()]

    (success, tiles_file, body_matrix,
        block, pre_crop_file) = fiji_grid_stitch(lnk_dir, lnk_cells,
                                                 lnk_path, False,
                                                 pre_copy_files=False,
                                                 tile_overlap=overlap)
    if not success:
        debug_log("Could not compute links to block", block_idx)
    else:
        inverse_d = { os.path.basename(fpath): coord
                     for coord, (fpath, cidx) in link_cells_d.items() }
        displs = TileConfigurator().parse(tiles_file)
        links_disp_d[block_idx] = { coord: displs[fname] for
                                   fname, coord in inverse_d.items() }

    return {'success': success, 'links': links_disp_d}


def list_grid_block_paths(n_rows, n_cols, row_cells, cell_size):

    cells_spec = [ make_groups(range(n_dim), cell_size)
                  for n_dim in (n_rows, n_cols) ]
    block_spec = [ [ (row_idx, col_idx),
                     [[row_cells[a][b] for b in col_idcs] for a in row_idcs] ]
                  for (row_idx, row_idcs), (col_idx, col_idcs)
                  in it.product(*cells_spec) ]

    return block_spec


def make_groups(elements, the_stride):

    groups = []
    n = len(elements)
    start_idx = 0
    group_idx = 0

    while start_idx < n:

        stride = the_stride
        would_remain = n - (start_idx + stride)

        if would_remain <= (the_stride/2):
            stride += would_remain

        groups.append((group_idx, elements[start_idx:(start_idx + stride)]))
        start_idx += stride
        group_idx += 1

    return groups


def get_link_blocks(blocks_dict):

    max_row, max_col = map(max, zip(*blocks_dict.keys()))
    link_blocks = {}

    get_link = lambda block_i, cell_i: blocks_dict[block_i][cell_i[0]][cell_i[1]]

    for (bl_i, bl_j), block_grid_files in blocks_dict.items():

        if (bl_i == max_row) or (bl_j == max_col):
            continue

        tl_block, tl_cell_idx = (bl_i, bl_j), (-1, -1)
        tr_block, tr_cell_idx = (bl_i, bl_j + 1), (-1, 0)
        bl_block, bl_cell_idx = (bl_i + 1, bl_j), (0, -1)
        br_block, br_cell_idx = (bl_i + 1, bl_j + 1), (0, 0)

        tl_link = get_link(tl_block, tl_cell_idx)
        tr_link = get_link(tr_block, tr_cell_idx)
        bl_link = get_link(bl_block, bl_cell_idx)
        br_link = get_link(br_block, br_cell_idx)

        link_blocks[(bl_i, bl_j)] = {tl_block: (tl_link, tl_cell_idx),
                                     tr_block: (tr_link, tr_cell_idx),
                                     bl_block: (bl_link, bl_cell_idx),
                                     br_block: (br_link, br_cell_idx)}

    return link_blocks


def compute_stitch_size(cell_size, displacements):

    img_height, img_width = cell_size
    n_rows, n_cols, row_cells = parse_image_grid_list(displacements.keys())[:3]

    cell_corner = np.r_[0, 0]
    sta_row = [ (cell_corner + displacements[fi])[1] for fi in row_cells[0] ]
    sta_col = [ (cell_corner + displacements[r[0]])[0] for r in row_cells ]

    cell_corner = np.r_[img_width, img_height]
    end_row = [ (cell_corner + displacements[fi])[1] for fi
               in row_cells[n_rows-1] ]
    end_col = [ (cell_corner + displacements[r[n_cols-1]])[0] for r
               in row_cells ]

    tl_corner = np.r_[min(sta_col), min(sta_row)]
    br_corner = np.r_[max(end_col), max(end_row)]
    c_size = tuple((br_corner - tl_corner).round().astype(int)[::-1])

    return c_size, tl_corner #, br_corner


class PairStitcher(object):

    def __init__(self, work_dir):

        self.work_dir = ensure_dir(work_dir)

    def __call__(self, identifier, partner_identifier, row_target, col_target,
                 scale, reduction_axis):

        scale_dir = ensure_dir(os.path.join(self.work_dir, str(scale)))
        next_scale_dir = ensure_dir(os.path.join(self.work_dir, str(scale+1)))
        temp_dir = ensure_dir(os.path.join(scale_dir, "work"))

        input_img = [identifier, partner_identifier]
        axis_name = ['rows', 'cols']
        print("Partners over {}: {} and {}".format(axis_name[reduction_axis],
                                                   os.path.basename(identifier),
                                                   os.path.basename(partner_identifier)))

        target_name = "{}_{}".format(row_target, col_target)
        target_file = "{}.{}".format(target_name, img_data_fmt)
        dir_stitch = setup_directories(os.path.join(temp_dir, target_name))
        stitched_img = os.path.join(next_scale_dir, target_file)

        if reduction_axis == 0:
            success = assemble_row(dir_stitch, input_img, stitched_img)
        else:
            success = assemble_column(dir_stitch, input_img, stitched_img)

        return success, stitched_img


def perform_assembly(directories, files, kind_tag,
                     assembly_fun, out_file_name):

    base_dir = os.path.dirname(out_file_name)
    base_name = os.path.basename(out_file_name)
    exec_fun = command_executor_factory(directories['log'], base_name)

    for _file in files:
        shutil.copy(_file, directories['in'])

    debug_log("Assembling of", kind_tag, "into", out_file_name, "on",
              directories['in'], "started")

    out_file, success = assembly_fun(exec_fun, {},
                                     directories['in'], directories['out'])
    if success:
        shutil.copyfile(out_file, out_file_name)
        shutil.rmtree(directories['in'])
        #shutil.rmtree(directories['out'])

    log_dir = os.path.join(base_dir, 'log')
    dir_util.copy_tree(directories['log'], log_dir)

    result_name = "done" if success else "failed"
    debug_log("Assembling of", out_file_name, result_name)

    return success


def process_command_line():

    description = "Performs stitching operations via command line"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("job_type", choices=job_types,
            help="What type of stitching operations to perform")
    parser.add_argument("files", nargs='*',
            help="Input image files to be stitched")
    parser.add_argument('-d', "--work-dir", default='/dev/shm',
            help="Where input, temporary, and output files will be written")
    parser.add_argument('-o', "--output-file", default='out.png',
            help="Pathname of output file")
    parser.add_argument('-t', "--threads", type=int,
                        default=max(1, mp.cpu_count() - 1),
            help=("Maximum number of simultaneous processes to execute"))
    parser.add_argument('-k', "--keep-work-dir", action="store_true",
            help="Avoids erasing of temporary work area")
    parser.add_argument('-v', "--tile-overlap", type=int,
                        default=25,
            help=("A-priori estimation of percentual area overlap between "
                  "contiguous tiles"))
    parser.add_argument('-r', "--also-save-reduced", type=int,
                        default=100,
            help=("If not equal to 100, saves a version of the image "
                  "scaled by the percentage specified"))
    parser.add_argument('-c', "--contrast", action="store_true",
            help="Perform simple contrast stretching of stitched image")

    args = parser.parse_args()

    if len(args.files) == 1:
        args.files = glob(args.files[0])

    return args


if __name__ == "__main__":
    sys.exit(main())
