
from glob import glob
import json
import multiprocessing as mp
import os.path as pth
import re

import cv2
import numpy as np

import tchebyshev_img as tc


def run_kind(kind, max_idx, base_dir=".", threads=1, roi=-1):
    files = glob(pth.join(base_dir, kind, "*.png"))
    with_z_pos = ( (get_file_z_pos(fi), fi) for fi in files )
    job_args = [ (fi, z, max_idx, base_dir, roi) for z, fi in with_z_pos ]
    if threads == 1:
        results = [ image_focus_worker(*args) for args in job_args ]
    else:
        pool = mp.Pool(processes=threads)
        jobs = [pool.apply_async(image_focus_worker, args) for args in job_args]
        pool.close()
        pool.join()
        results = [ job.get() for job in jobs ]
    curve = sorted(results)
    out_file = pth.join(pth.abspath(base_dir), kind, str(roi), "focus-curve-moment-{}.json".format(max_idx))
    with open(out_file, 'w') as fobj:
        json.dump(curve, fobj)
    print("{} curve written to {}".format(kind, out_file))

    return curve


def get_file_z_pos(filename):
    base = pth.splitext(pth.basename(filename))[0]
    num = re.findall(r'[mp][0-9]+', base)
    if len(num) == 0:
        raise ValueError("File has no valid Z position: {}".format(filename))
    n_str = num[0]
    sign = -1 if n_str[0] == 'm' else 1 

    return sign*int(n_str[1:])


def image_focus_worker(filename, z_pos, max_idx, base_dir, center_roi):
    img_rgb = cv2.imread(filename)
    img = ( img_rgb if center_roi == -1 else crop_center_chunk(img_rgb, center_roi) )
    img_norm = tc.normalize_rgb_image(img)
    N, M = img_norm.shape
    moment_generator = tc.ImageMomentGenerator(N, M, base_dir)
    moments = moment_generator.generate_moments(img_norm, max_idx)

    return (z_pos, tc.focus_measure(moments))


"""
The next two functions are pasted from register_stack_dft_mp.py
"""
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


if __name__ == "__main__":
    import sys
    kinds = ['ppl', 'xpl']
    max_idx = int(sys.argv[1])
    roi = int(sys.argv[2])
    { kind: run_kind(kind, max_idx, threads=6, roi=roi) for kind in kinds }

