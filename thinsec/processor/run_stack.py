#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
from glob import iglob
import itertools as it
import multiprocessing as mp
import os
import os.path as pth
import re
import shutil
import sys
import tempfile

import numpy as np

from archivemount import ArchiveMount
from common import (DebugLog, img_data_fmt, parse_image_grid_list,
                    ensure_dir, shell_retcode)
from cv_tools import (cv2_to_file, image_resize, file_to_cv2)
import deepzoom
from divide_correction import (apply_blankfield, generate_write_blankfield)
from fiji_driver import (TileConfigurator, fiji_grid_fuse)
from register_stack_xp import register_images
from stitcher import assemble_blocklink
from sweep_processor import setup_directories

debug_log = DebugLog()
max_threads = max(1, mp.cpu_count())

kinds = {"xpl", "ppl"}


def main():

    opt = process_command_line()
    print opt

    output_dir = pth.abspath(opt.output_dir)
    ensure_dir(output_dir)

    ensure_dir(opt.work_dir)
    stack_name = pth.basename(opt.stack_base)
    work_dir = tempfile.mkdtemp(prefix=stack_name + '-',
                                dir=opt.work_dir)

    disk_work_dir = ensure_dir(opt.disk_work_dir)

    # Reference cell grid for registration and stitching computation
    ref_kind, ref_angle = 'ppl', 0

    src_dirs, tgt_dirs, tgt_base_dir = set_stack_structure(opt.stack_base,
                                                           output_dir,
                                                           opt.preregister)
    stack_config, s_rows, s_cols = determine_stack_config(src_dirs)

    # 0. Register cells across all angles of all image kinds
    if opt.preregister:
        success = align_cells(stack_config, src_dirs, tgt_dirs, ref_kind, ref_angle,
                              work_dir, max_threads, opt.crop_size, opt.ppl_blankfield_file)
    else:
        print "Cells pre-registration skipped"
        success = skip_align_cells(opt.stack_base, stack_config, tgt_base_dir)

    result = "done" if success else "failed"
    debug_log("Registration job", result)

    if success:

        work_pending = {kind: cfg.keys() for kind, cfg in stack_config.items()}

        # 1. Apply blankfield correction to PPL images
        for ppl_angle in work_pending['ppl']:
            ppl_angle_dir = pth.join(tgt_base_dir, "ppl", str(ppl_angle))
            blankfield = pth.join(ppl_angle_dir, "blankfield-stat.png")

            if opt.ppl_blankfield_file is None:
                images = list(iglob(pth.join(ppl_angle_dir, "??_??.png")))
                if not pth.isfile(blankfield):
                    print "Computing blankfield for PPL image angle", ppl_angle
                    generate_write_blankfield(images, ppl_angle_dir,
                                              threads=opt.threads)
                else:
                    print "Blankfield for PPL image angle", ppl_angle, "exists"
            else:
                if not opt.preregister:
                    ppl_bfield = pth.expanduser(opt.ppl_blankfield_file)
                    shutil.copy(ppl_bfield, blankfield)
                    print "Copied blankfield for PPL image angle", ppl_angle, "from", ppl_bfield, "into", blankfield

        # 2. Find stitch displacements from the reference cell grid
        displacements_savefile = pth.join(tgt_base_dir, "stack-displacements.txt")
        tc = TileConfigurator()
        if pth.isfile(displacements_savefile):
            displacements = tc.parse(displacements_savefile)
            success = True
        else:
            files_in = [ pth.join(tgt_dirs[ref_kind], str(ref_angle), fname)
                        for row in stack_config[ref_kind][ref_angle]['row_cells']
                        for fname in row ]
            work_dirs = setup_directories(work_dir)
            success, _, displacements = assemble_blocklink(work_dirs, files_in, None,
                                                           keep_files=opt.keep_work_dir,
                                                           threads=opt.threads,
                                                           do_crop=False, do_fuse=False)
            tc.generate(displacements, displacements_savefile)

        # 3. Apply stitch displacements to all angles of all image kinds
        work_order = imerge(*[ [ (kind, ang) for ang in angles ]
                              for kind, angles in work_pending.items() ])
        disk_work_dirs = setup_directories(disk_work_dir)

        if opt.ppl_blankfield_file is not None:
            blur = None if opt.preregister else -1
        else:
            blur = 0.6

        job_args = [ (tgt_dirs[kind], angle, displacements, disk_work_dirs, kind,
                      opt.post_register, blur,
                      [ pth.join(tgt_dirs[kind], str(angle), fname) for row
                       in stack_config[kind][angle]['row_cells'] for fname in row ])
                    for kind, angle in work_order ]

        if opt.threads == 1:
            files = [ apply_displacements(*args) for args in job_args ]
        else:
            pool = mp.Pool(processes=opt.threads)
            jobs = [ pool.apply_async(apply_displacements, args) for args in job_args ]
            pool.close()
            pool.join()
            files = [ job.get() for job in jobs ]

        if not opt.keep_work_dir:
            shutil.rmtree(work_dir)

        # 4. Register the images for removing mismatch due to birrefringence
        target_dir = pth.join(tgt_base_dir, "registered")
        if opt.post_register:
            files_map = { fi: re.sub(r"([xp]pl)/full", r"registered/\1", fi)
                         for fi in files }

            if not all(pth.isfile(fo) for fo in files_map.values()):
                register_images(files, 4096, opt.threads,
                                ensure_dir(target_dir), True, False,
                                target_files_map=files_map,
                                make_jpeg=opt.write_also_jpeg)

            job_args_gen = ( (fi, pth.splitext(pth.basename(fo))[0]) for
                            fi, fo in files_map.items() )
            job_args = [ (fi,
                          pth.join(target_dir, "pyr-{}.tar".format(fo)),
                          "{}.dzi".format(fo)) for fi, fo in job_args_gen ]
            if opt.threads == 1:
                success = [ make_pyramid(*args) for args in job_args ]
            else:
                pool = mp.Pool(processes=opt.threads)
                jobs = [ pool.apply_async(make_pyramid, args) for args in job_args ]
                pool.close()
                pool.join()
                success = [ job.get() for job in jobs ]

            origs = ( arc + ".orig" for _, arc, _ in job_args )
            [ os.remove(orig) for orig in origs if pth.exists(orig) ]

        # A flag file is a dirty way of notifying other processes that this
        # job is complete
        flag_file = pth.join(tgt_base_dir, "done")
        with open(flag_file, 'a'): pass     # Touch file

    result = "done" if success else "failed"
    debug_log("Stack job", result)

    return shell_retcode(success)


def set_stack_structure(stack_name, work_dir, make_subdirs=True):

    st_name = pth.abspath(stack_name)
    source_dirs = { kind: pth.join(st_name, kind) for kind in kinds }

    base_dir = ensure_dir(pth.join(work_dir, pth.basename(st_name)))
    target_dirs = { kind: pth.join(base_dir, kind) for kind in kinds }

    if make_subdirs and all( pth.exists(src_dir) for src_dir in
                            source_dirs.values() ):
        [ensure_dir(tdir) for tdir in target_dirs.values()]

    return source_dirs, target_dirs, base_dir


def determine_stack_config(source_dirs):

    cell_match = re.compile(r'^([0-9]+)/([0-9_]+.{})'.format(img_data_fmt))
    stack_config = {}

    cwd = os.getcwd()
    for kind in kinds:

        # Gather properties of all angles of this kind
        os.chdir(source_dirs[kind])
        all_images = iglob("*/*." + img_data_fmt)
        is_cells = ( cell_match.search(fpath) for fpath in all_images )
        only_cells = ( match.groups() for match in is_cells if match )
        cell_list_d = { int(ang): [fpath for _, fpath in cells] for ang, cells
                       in it.groupby(only_cells, lambda e: e[0])}
        cell_cfg_d = { ang: parse_image_grid_list(cl, True) for ang, cl
                      in cell_list_d.items() }

        # Check that properties are consistent across all angles
        num_rows = set( prop['n_rows'] for prop in cell_cfg_d.values() )
        assert len(num_rows) == 1, ("Inconsistent amount of rows "
                                    "in kind {}: {}".format(kind, num_rows))
        num_cols = set( prop['n_cols'] for prop in cell_cfg_d.values() )
        assert len(num_cols) == 1, ("Inconsistent amount of columns "
                                    "in kind {}: {}".format(kind, num_cols))

        stack_config[kind] = cell_cfg_d

    # Check that the grid is the same for all image kinds
    g_rows, g_cols = zip(*[ (cfg[0]['n_rows'], cfg[0]['n_cols']) for kind, cfg
                           in stack_config.items() ])
    gs_rows, gs_cols = set(g_rows), set(g_cols)
    assert len(gs_rows) == 1, ("Inconsistent amount of rows "
                                   "across kinds: {}".format(gs_rows))
    assert len(gs_cols) == 1, ("Inconsistent amount of columns "
                                   "across kinds: {}".format(gs_cols))
    os.chdir(cwd)

    return stack_config, gs_rows.pop(), gs_cols.pop()


def align_cells(stack_config, src_dirs, tgt_dirs, ref_kind, ref_angle, work_dir,
                threads, crop_size, blankfield_file=None):

    if blankfield_file is not None:
        bfields_dir = tempfile.mkdtemp(dir="/tmp", prefix="bfield")

    job_args = []

    for row in stack_config[ref_kind][ref_angle]['row_cells']:
        for fname in row:
                
            if blankfield_file is None:
                s_dirs = src_dirs
            else:
                kind = 'ppl'
                ppl_stage_dir = ensure_dir(pth.join(bfields_dir, fname))
                ppl_in_dir = ensure_dir(pth.join(ppl_stage_dir, "in"))
                ppl_out_dir = ensure_dir(pth.join(ppl_stage_dir, "out"))
                ppl_files = { pth.join(src_dirs[kind], str(ang), fname):
                              pth.join(ppl_in_dir, str(ang) + fname)
                              for ang in stack_config[kind].keys() }
                [ shutil.copyfile(src, dst) for src, dst in ppl_files.items() ]
                apply_blankfield(ppl_files.values(), blankfield_file, ppl_stage_dir,
                                 blur_alpha=-1, threads=max_threads)
                [ ensure_dir(pth.join(ppl_out_dir, str(ang))) for ang
                 in stack_config[kind].keys() ]
                [ shutil.move(pth.join(ppl_stage_dir, str(ang) + fname),
                              pth.join(ppl_out_dir, str(ang), fname))
                              for ang in stack_config[kind].keys() ]
                s_dirs = {'ppl': ppl_out_dir, 'xpl': src_dirs['xpl']}

            io_map = { pth.join(s_dirs[kind], str(ang), fname):
                        pth.join(tgt_dirs[kind], str(ang), fname)
                          for kind in stack_config.keys()
                          for ang in stack_config[kind].keys() }
            args = (io_map.keys(), crop_size, 1, work_dir)
            kwargs = dict(use_borders=True, first_image_is_absolute=False,
                          make_center_chunk=False, target_files_map=io_map)
            job_args.append((args, kwargs))

    pool = mp.Pool(processes=threads)
    jobs = [ pool.apply_async(register_images, args, kwargs) for args, kwargs
            in job_args ]
    pool.close()
    pool.join()
    success = all( job.get() for job in jobs )

    if blankfield_file is not None:
        shutil.rmtree(bfields_dir)

    return success


def skip_align_cells(stack_base, stack_config, target_base_dir):

    try:
        for kind, kind_data in stack_config.items():
            kind_out_dir = ensure_dir(pth.join(target_base_dir, kind))
            for angle, config in kind_data.items():
                angle_in_dir = pth.join(stack_base, kind, str(angle))
                angle_out_dir = ensure_dir(pth.join(kind_out_dir, str(angle)))
                cell_list = sum(config['row_cells'], [])
                in_cells = [ pth.join(angle_in_dir, c) for c in cell_list ]
                out_cells = [ pth.join(angle_out_dir, c) for c in cell_list ]
                for in_cell, out_cell in zip(in_cells, out_cells):
                    if not pth.islink(out_cell):
                        os.symlink(in_cell, out_cell)
        success = True
    except OSError:
        debug_log("Error making symlinks!")
        success = False

    return success


def apply_displacements(target_dir, angle, displacements, work_dirs,
                        kind, do_register, blur, files_in):

    file_out = os.path.join(target_dir, "full-{}.{}".format(angle, img_data_fmt))
    if not pth.isfile(file_out):
        kind_dir = ensure_dir(pth.join(work_dirs['temp'], kind))
        my_work_dirs = setup_directories(pth.join(kind_dir, str(angle)))

        if blur is not None:
            new_dir = blankfield_guard(files_in, kind, my_work_dirs['in'], blur)
        else:
            new_dir = pth.dirname(files_in[0])

        coords_fi = pth.join(new_dir, "tiles.txt")
        TileConfigurator().generate(displacements, coords_fi)
        success, img, _ = fiji_grid_fuse(my_work_dirs, coords_fi, file_out,
                                         do_crop=False)
        shutil.rmtree(my_work_dirs['in'])
    else:
        debug_log("Ensemble", file_out, "already exists.")
        img = None

    cropped_fpath = os.path.join(target_dir, "center-crop-{}.jpg".format(angle))
    if not pth.isfile(cropped_fpath):
        load_img = img if img is not None else file_to_cv2(file_out)
        cropped_img = crop_center_chunk(load_img, 1024)
        cv2_to_file(cropped_img, cropped_fpath)
    else:
        debug_log("Center-cropped chunk", cropped_fpath, "already exists.")

    reduced_fpath = os.path.join(target_dir, "full-{}.small.jpg".format(angle))
    if not pth.isfile(reduced_fpath):
        load_img = img if img is not None else file_to_cv2(file_out)
        reduced_img = image_resize(load_img, 30)
        cv2_to_file(reduced_img, reduced_fpath)
    else:
        debug_log("Reduced size image", reduced_fpath, "already exists.")

    if not do_register:
        pyramid_archive = pth.join(target_dir, "pyramid-{}.tar".format(angle))
        if not pth.isfile(pyramid_archive):
            make_pyramid(img, pyramid_archive, "{}-{}.dzi".format(kind, angle))
        else:
            debug_log("Pyramid archive", pyramid_archive, "already exists.")

    return file_out 


def blankfield_guard(files_in, kind, work_dir, blur, **kwargs):

    files_in_dir = pth.dirname(files_in[0])

    if kind == 'ppl':
        blankfield = pth.join(files_in_dir, "blankfield-stat.png")
        debug_log("Applying blankfield image", blankfield, "into", work_dir)
        apply_blankfield(files_in, blankfield, work_dir,
                         blur_alpha=kwargs.get('blur_alpha', blur),
                         threads=kwargs.get('threads', 1))
        new_dir = work_dir
    else:
        new_dir = files_in_dir

    return new_dir


def make_pyramid(image_in, pyramid_archive_path, descriptor_name):

    temp_mount = tempfile.mkdtemp(dir='/tmp')

    with ArchiveMount(pyramid_archive_path, temp_mount) as pyramid_contain_dir:

        pyramid_descriptor = pth.join(pyramid_contain_dir,
                                      descriptor_name)
        creator = deepzoom.ImageCreator(tile_size=128, tile_overlap=2,
                                        tile_format="jpg", image_quality=0.95,
                                        resize_filter="bicubic")
        load_img = ( image_in if isinstance(image_in, np.ndarray)
                    else file_to_cv2(image_in) )
        creator.create(load_img, pyramid_descriptor)

    debug_log("Pyramid archive", pyramid_archive_path, "created.")

    #os.rmdir(temp_mount)
    shutil.rmtree(temp_mount)

    return True


def crop_center_chunk(image, chunk_length=1024):

    center = np.array(image.shape[:2][::-1])/2
    center_crop_delta = np.r_[chunk_length, chunk_length]/2

    corner_start = center - center_crop_delta
    corner_end = center + center_crop_delta

    bounding_box = [corner_start, corner_end]
    min_x, max_x = sorted(co[0] for co in bounding_box)
    min_y, max_y = sorted(co[1] for co in bounding_box)

    im_roi = image[min_y:max_y, min_x:max_x]

    return im_roi


def imerge(a, b):
    for i, j in it.izip(a, b):
        yield i
        yield j


def process_command_line():

    description = "Processes a stack of cells of PPL and XPL images"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("stack_base",
            help=("Path to the stack's -ppl and -xpl directories, without "
                  "either of the suffixes"))
    parser.add_argument("output_dir",
            help="Where to save all processed images")
    parser.add_argument('-d', "--work-dir", default='/dev/shm',
            help="Where input, temporary, and output files will be written")
    parser.add_argument('-i', "--disk-work-dir", default=pth.expanduser('~/disktemp'),
            help="Where input, temporary, and output files will be written")
    parser.add_argument('-c', "--crop-size", type=int, default=2048,
            help="image crop side length for translation estimation")
    parser.add_argument('-t', "--threads", type=int,
                        default=max_threads,
            help=("Maximum number of simultaneous processes to execute"))
    parser.add_argument('-b', "--ppl-blankfield-file", default=None,
            help="Path to a blankfield image file for PPL images")
    parser.add_argument('-k', "--keep-work-dir", action="store_true",
            help="Avoids erasing of temporary work area")
    parser.add_argument('-p', "--preregister", action="store_false",
            help="Skip the cell pre-registration procedure")
    parser.add_argument('-o', "--post-register", action="store_true",
            help="Perform registration procedure of stitched ensembles")
    parser.add_argument('-r', "--also-save-reduced", type=int,
                        default=100,
            help=("If not equal to 100, saves a version of the image "
                  "scaled by the percentage specified"))
    parser.add_argument('-w', "--with-center-crop", action="store_true",
            help="If set, saves a 1024x1024 crop at the center of the image")
    parser.add_argument('-j', "--write-also-jpeg", action='store_true',
            help="Create JPEG versions of the registered images")

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
