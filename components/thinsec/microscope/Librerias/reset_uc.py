# -*- coding: utf-8 -*-

import os
import os.path as pth
import gevent
import gevent_inotifyx as inotify
from gevent.queue import Queue
import serial


def reset_controller(port, reset_pattern):
    port_name = port.port
    port_rate = port.baudrate
    event_waiter = FileCreateWaiter(port_name)
    port.write(reset_pattern)
    port.close()
    del port
    event_waiter.wait()
    gevent.sleep(0.05)
    port_new = serial.Serial(port_name, port_rate)

    return port_new


class FileCreateWaiter(object):
    def __init__(self, file_path):
        self.path = file_path
        self.parent = pth.dirname(pth.abspath(self.path))
        self.q = Queue()
        self.fd = inotify.init()

    def event_producer(self):
        while True:
            events = inotify.get_events(self.fd)
            for event in events:
                self.q.put(event)
                if event.name == pth.basename(self.path):
                    return

    def wait(self):
        try:
            inotify.add_watch(self.fd, self.parent, inotify.IN_CREATE)
            eventer = gevent.spawn(self.event_producer)
            while True:
                event = self.q.get()
                if event.name == pth.basename(self.path):
                    break
        finally:
            gevent.wait((eventer,))
            os.close(self.fd)

        return True


if __name__ == "__main__":
    import serial
    uart = serial.Serial("/dev/ttyACM0", 115200)
    uart = reset_controller(uart, chr(0x0A) + '1')
    print ("After resetting:", uart.read(1) + uart.read(uart.inWaiting()))
