#!/usr/bin/env python2

import calendar
import collections
from datetime import datetime
import functools
import linecache
import math
import os
import os.path as pth
import sys
import time

from numpy import r_


def get_utc_now_ms():

    now = datetime.utcnow()
    epoch = calendar.timegm(now.utctimetuple())
    ms = now.microsecond/1000
    return epoch*1000 + ms


def log_line(*args, **kwargs):

    newline = ' ' if ('newline' in kwargs and not kwargs['newline']) else '\n'

    now = get_utc_now_ms()
    line = "{}: {}{}".format(integer_separate(now),
                             ' '.join(map(str, args)), newline)
    return line


class DebugLog(object):

    def __init__(self, use_stderr=False):

        self.last = get_utc_now_ms()
        self.stream = sys.stderr if use_stderr else sys.stdout

    def __call__(self, *args, **kwargs):

        newline = ' ' if ('newline' in kwargs and not kwargs['newline']) else '\n'

        now = get_utc_now_ms()
        delta = now - self.last
        self.last = now

        line = "+{} ms: {}{}".format(integer_separate(delta),
                                     ' '.join(map(str, args)), newline)
        self.stream.write(line)


def print_state_change(e):

    print 'event: %s, src: %s, dst: %s' % (e.event, e.src, e.dst)


def integer_separate(number):

    r_st = str(number)[::-1]
    sep_r = ' '.join(r_st[3*i:3*(i+1)] for i in range(1+len(r_st)/3))

    return sep_r[::-1].strip()


def get_digits(number):

    return max(1, int(math.ceil(math.log10(number))))


def ensure_dir(save_dir, mode=0750):

    if not pth.isdir(save_dir):
        try:
            os.mkdir(save_dir, mode)
        except OSError:
            raise OSError("Unable to create folder", save_dir)

    return save_dir


def spawn_subdir(base_dir, subdir):

    new_dir = pth.join(base_dir, subdir)
    ensure_dir(base_dir)
    ensure_dir(new_dir)

    return new_dir


def PrintException():

    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


class EdgeDetector(object):

    def __init__(self, start_val=False, trigger_edge=True, edge_callback=None, cooloff=0):

        self.curr = self.last = start_val

        self.__edge = trigger_edge
        self.__event = False
        self.callback = edge_callback

        self.cooloff = cooloff
        self.last_t = time.time()

    def __call__(self, value=None):

        if value is None:
            return self.asserted
        else:
            self.assign(value)

    @property
    def asserted(self):

        now = time.time()
        dt = now - self.last_t

        if dt <= self.cooloff:
            return self.__event
        else:
            return False

    def assign(self, value):

        self.curr = value

        now = time.time()
        dt = now - self.last_t
        is_cool = (dt >= self.cooloff)

        trigger = self.curr ^ self.last
        level = not (self.curr ^ self.__edge)

        if trigger and level:
            if is_cool:
                self.last_t = now
                self.__event = trigger
                self.callback()

        self.last = self.curr

    def clear(self):

        self.__event = False


def parse_image_grid_list(files_in, as_dict=False):

    files = sorted(files_in)

    file_table = [(f, pth.splitext(pth.basename(f))[0].split('_'))
                  for f in files]
    as_ints = [map(int, f[1]) for f in file_table]
    n_cols = 1 + max(c[1] for c in as_ints) - min(c[1] for c in as_ints)
    n_rows = 1 + max(c[0] for c in as_ints) - min(c[0] for c in as_ints)
    assert n_rows * n_cols == len(files), "Missing input images for a full stitch in {}, {}: {}".format(n_rows, n_cols, files)

    extensions = set(pth.splitext(pth.basename(f))[1] for f in files)
    assert len(extensions) == 1, "There is more than one image file type: {}".format(' '.join(extensions))
    extension = extensions.pop()[1:]

    all_digits_row, all_digits_col = zip(*[map(len, f[1]) for f in file_table])
    assert max(all_digits_row) == min(all_digits_row), "File naming is not consistent over rows"
    assert max(all_digits_col) == min(all_digits_col), "File naming is not consistent over columns"
    digits_row = min(all_digits_row)
    digits_col = min(all_digits_col)

    rows = sorted(list(set(f[1][0] for f in file_table)))
    row_cells = [ [f[0] for f in file_table if f[1][0] == row]
                 for row in rows ]

    if as_dict:
        return {'n_rows': n_rows, 'n_cols': n_cols, 'row_cells': row_cells,
                'extension': extension, 'rows': rows, 'digits_row': digits_row,
                'digits_col': digits_col}

    else:
        return n_rows, n_cols, row_cells, extension, rows, digits_row, digits_col


################################################################################
#  Image handling and processing variables                                     #
################################################################################

img_data_fmt = 'png'    #'jpg'


def image_array_crop(image, crop):

    height, width = image.shape[:2]

    denorm = r_[width, height, width, height].astype(float)
    rectangle = (r_[crop]*denorm).round().astype(int)

    cropped = image[rectangle[1]:rectangle[3],
                    rectangle[0]:rectangle[2]]

    return cropped


################################################################################
#  Caching
################################################################################

def lru_cache (maxsize=128):
    '''Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    '''
    def decorating_function (user_function):
        cache = collections.OrderedDict()    # order: least recent to most recent

        @functools.wraps(user_function)
        def wrapper(*args, **kwds):
            key = args
            if kwds:
                key += tuple(sorted(kwds.items()))
            try:
                result = cache.pop(key)
                wrapper.hits += 1

            except KeyError:
                result = user_function(*args, **kwds)
                wrapper.misses += 1
                if len(cache) >= maxsize:
                    cache.popitem(0)    # purge least recently used cache entry

            cache[key] = result         # record recent use of this key
            return result

        wrapper.hits = wrapper.misses = 0
        return wrapper

    return decorating_function


class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            raise KeyError("Key {} not found".format(key))

    def __setitem__(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __delitem__(self, key):
        try:
            self.cache.pop(key)
        except KeyError:
            raise KeyError("Key {} not found".format(key))

    def __len__(self):
        return len(self.cache)

    def clear(self):
        return self.cache.clear()


def shell_retcode(success):
    # Shell return code convention is the opposite of Python's bool-int mapping
    return int(not success)
