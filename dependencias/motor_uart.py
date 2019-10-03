# -*- coding: utf-8 -*-

import queue
import struct
import serial
import motor_api
from reset_uc import reset_controller


class MotorLink(motor_api.BaseMotorLink):
    def __init__(self, port, baudrate, debug=False):
        motor_api.BaseMotorLink.__init__(self)
        self.port_name = port
        self.port_rate = baudrate
        self.uart = serial.Serial(self.port_name, self.port_rate)
        self.uart_reply_queue = queue.Queue()
        self.uart_info_queue = queue.Queue()
        self.uart_debug_queue = queue.Queue()
        self.readout_run = False
        self.debug = debug

    def global_reset(self):
        # Using the Due's native USB port, which does not perform reset
        # the uC upon port opening.
        self.uart_flush()
        sequence = chr(motor_api.global_register_offset["global-reset"]) + "1"
        self.uart = reset_controller(self.uart, sequence)

    def motor_command_byte(self, motor, command, value):
        to_send = struct.pack("<BB", self.register_address(motor, command), int(value))
        self.send(to_send)

    def motor_command_word(self, motor, command, value):
        to_send = struct.pack("<Bh", self.register_address(motor, command), int(value))
        self.send(to_send)

    def motor_command_int(self, motor, command, value):
        to_send = struct.pack("<Bi", self.register_address(motor, command), int(value))
        self.send(to_send)

    def global_command_byte(self, command, value):
        to_send = struct.pack(
            "<BB", motor_api.global_register_offset[command], int(value)
        )
        self.send(to_send)

    def global_command_word(self, command, value, unsigned=False):
        word_spec = ["h", "H"]
        spec = "<B" + word_spec[unsigned]
        to_send = struct.pack(
            spec, motor_api.global_register_offset[command], int(value)
        )
        self.send(to_send)

    def global_command_sequence(self, command, values):
        length = len(values)
        if length <= 32:
            spec = "<BB" + "H" * 32
            sequence = map(int, values) + [0] * (32 - length)
            to_send = struct.pack(
                spec, motor_api.global_register_offset[command], length, *sequence
            )
            self.send(to_send)
        else:
            print("Sequences longer than 32 values will not be sent out")

    def send(self, to_send):
        if self.debug:
            print("UART send:", ", ".join("{:02X}".format(ord(c)) for c in to_send))
        print("UART send:", ", ".join("{:02X}".format(ord(c)) for c in to_send))
        self.uart.write(to_send)
        self.check_command_reply()

    def check_command_reply(self):
        reply = self.receive()
        assert reply == "ok", "Bad reply from controller: {}".format(reply)

    def restart(self):
        self.global_reset()
        reply = self.uart.read(1) + self.uart.read(self.uart.inWaiting())
        assert reply == "st", "Bad reply from controller: {}".format(reply)
        print("Motor controller restarted.")

    def receive(self, block=True):
        assert self.readout_run or block, "UART is not being read"
        try:
            rec = self.uart_reply_queue.get(block=block)
        except Queue.Empty:
            rec = ""

        print("UART recieve:", rec)
        return rec

    def uart_readout_loop(self):
        self.uart_flush()
        self.readout_run = True
        if self.debug:
            print("reading UART")
        while self.readout_run:
            # Smallest expected message is 2-byte long
            read = self.uart.read(2)
            if self.debug:
                read_hex = " ".join("{:02x}".format(ord(x)) for x in read)
                print("UART rec: {} ({})".format(read_hex, read))
            if read == "do":
                self.uart_info_queue.put(read)
            elif read.startswith("m"):
                # Motor position packets include a 32-bit int
                read += self.uart.read(4)
                self.uart_info_queue.put(read)
            elif read.startswith("u"):
                read += self.uart.read(1)
                self.uart_debug_queue.put(read)
            else:
                self.uart_reply_queue.put(read)

    def uart_flush(self):
        self.uart.timeout = 0.1
        self.uart.read(1)
        self.uart.read(self.uart.inWaiting())
        self.uart.timeout = None
