#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os.path as pth
import threading

import pyinotify


class FilesWaiter(object):

    def __init__(self, file_paths, watch_manager):
        
        paths = [file_paths,] if isinstance(file_paths, str) else file_paths
        self.file_p = {pth.abspath(fp): threading.Event() for fp in paths}

        self.watcher = FilesCloseWatcher(self.file_p)
        self.watch_manager = watch_manager
        self.notifier = pyinotify.ThreadedNotifier(watch_manager, self.watcher)
        self.notifier.start()
        base_path = pth.commonprefix([ pth.dirname(fp) for fp in
                                      self.file_p.keys() ])
        self.watch = watch_manager.add_watch(base_path,
                                             pyinotify.IN_CLOSE_WRITE, rec=True)
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):

        [signal.wait() for signal in self.file_p.values()]
        self.watch_manager.rm_watch(self.watch.values())
        self.notifier.stop()

    @property
    def are_done(self):
        return all(signal.is_set() for signal in self.file_p.values())

    def wait(self):

        self.thread.join()
        return self.are_done


class FilesCloseWatcher(pyinotify.ProcessEvent):

    def __init__(self, file_paths_events):

        pyinotify.ProcessEvent.__init__(self)
        self.signal_paths_events = { pth.abspath(fp): signal for fp, signal in
                                    file_paths_events.items() }

    def process_IN_CLOSE_WRITE(self, event):

        abspath = pth.abspath(event.pathname)
        if abspath in self.signal_paths_events:
            self.signal_paths_events[abspath].set()
