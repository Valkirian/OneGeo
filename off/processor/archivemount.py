#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import subprocess as sp
import os
import os.path as pth

import pyinotify

from common import ensure_dir
from ioevent import FilesWaiter


class ArchiveMount(object):

    def __init__(self, archive_path, base_mount_dir):

        self.archive_path = archive_path

        self.mount_dir = pth.join(base_mount_dir,
                                  "mnt-" + pth.basename(archive_path))
        self.mount_available = False

        if pth.isdir(self.mount_dir):
            raise OSError("Mount directory {} already exists".format(self.mount_dir))
        ensure_dir(self.mount_dir)
        self.mount_available = True 

    def __enter__(self):

        if not self.mount_available:
            raise OSError("Mountpoint {} is not yet available".format(self.mount_dir))
        
        # Create the target archive
        with open(self.archive_path, 'a'): pass

        wm = pyinotify.WatchManager()
        self.arc_waiter = FilesWaiter(self.archive_path, wm)

        retcode = sp.check_call(["archivemount", self.archive_path, self.mount_dir,
                                 "-o", "nobackup,big_writes"])
        if retcode != 0:
            raise OSError(("Could not mount archive {} on "
                           "mountpoint {}").format(self.archive_path, self.mount_dir))
        self.orig_file = self.archive_path + ".orig"
        if pth.exists(self.orig_file):
            os.remove(self.orig_file)

        return self.mount_dir

    def __exit__(self, exc_type, exc_value, traceback):

        retcode = sp.check_call(["fusermount", "-u", self.mount_dir])
        if retcode != 0:
            raise OSError(("Could not unmount mountpoint {}").format(self.mount_dir))

        self.arc_waiter.wait()
        os.rmdir(self.mount_dir)
        if pth.exists(self.orig_file):
            os.remove(self.orig_file)
