import asyncore
import sys

import cv2
import numpy as np
import pyinotify


def main():

    source_path = sys.argv[1]

    wm = pyinotify.WatchManager()
    notifier = pyinotify.AsyncNotifier(wm, EventHandler())
    wdd = wm.add_watch(source_path, pyinotify.IN_CLOSE_WRITE)

    try:
        asyncore.loop()
    except KeyboardInterrupt:
        pass


class EventHandler(pyinotify.ProcessEvent):

    def __init__(self, *args, **kwargs):

        pyinotify.ProcessEvent.__init__(self, *args, **kwargs)
        self.stats_name = ['max', 'min', 'mean', 'sum']
        self.stats = dict(zip(self.stats_name, [0]*len(self.stats_name)))

    def process_IN_CLOSE_WRITE(self, event):

        new_stats = self.get_stats(event.path)
        change = {key: new_stats[key] - self.stats[key]
                  for key in self.stats_name}
        self.stats = new_stats

        msg_new = ("New: Max {0[max]}; Min {0[min]}; Mean {0[mean]}; "
                   "Sum {0[sum]}")
        msg_chg = ("Chg:  Max {0[max]:+d}; Min {0[min]:+d}; Mean {0[mean]:+d}; "
                   "Sum {0[sum]:+d}")

        print("\n")
        print(msg_new.format(new_stats))
        print(msg_chg.format(change))


    def get_stats(self, source_path):

        img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        area = np.prod(img.shape[:2])
        sum = 100*(img.astype(float)/area).sum()
        stats_out = [img.max(), img.min(), round(100*img.mean()), sum]
        stats = dict(zip(self.stats_name, map(int, stats_out)))

        return stats


if __name__ == "__main__":
    main()
