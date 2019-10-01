import pyinotify
import time
from fiji_driver import FilesWaiter

files = ['/dev/shm/algo_{}.txt'.format(i) for i in range(3)]
wm = pyinotify.WatchManager()
watches = FilesWaiter(files, wm)
time.sleep(10)
print "Are done?", watches.are_done
print watches.wait()
