#import telnetlib

config_base = {'x': {'name': "X axis",
                     'index': 1},
               'y': {'name': "Y axis",
                     'index': 2},
               'z': {'name': "Z axis",
                     'index': 3},
             'pol': {'name': "Polarizer",
                     'index': 4},
             'ana': {'name': "Analyzer",
                     'index': 5}}

motor_address = { axis: setting['index']*0x10 for axis, setting in
                  config_base.items() }
motor_register_offset = {'mode': 0x00, 'dir': 0x01, 'steps': 0x02,
                         'period': 0x03, 'steptime': 0x04, 'hold': 0x05,
                         'acceleration': 0x06, 'speed-max': 0x07,
                         'position-current': 0x08, 'position-target': 0x09,
                         'speed-constant': 0x0A, 'backlash': 0x0B,
                         'speed-monitor': 0x0D, 'dir-invert': 0x0E}

motor_modes = {'FULL': 0, 'HALF': 1, 'QUARTER': 2,
               'EIGTHTH': 3, 'SIXTEENTH': 4}

global_register_offset = {'motor-status': 0x01, 'led-state': 0x02,
                          'led-fan-duty': 0x03, 'cam-fan-duty': 0x04,
                          'sequence-cmd': 0x05, 'sequence-steps': 0x06,
                          'cam-shutter-time': 0x07, 'sequence2-steps': 0x08,
                          'cam-wait-time': 0x09, 'global-reset': 0x0A, 
                          'led2-state': 0x0C, 'plate-o-ana': 0x10, 'plate-i-ana': 0x11,
                           'plate-o-pol': 0x10, 'plate-i-pol': 0x21}


def get_hardware_config():
    modes = []
    indices = []
    for mode, idx in get_modes():
        modes.append(mode.capitalize())
        indices.append(idx)
    stepmodes = {'modes': modes, 'indices': indices}
    config = config_base
    for axis in config.keys():
        config[axis]['stepmodes'] = stepmodes

    return config


def get_modes():
    return motor_modes.items()


class BaseMotorLink(object):
    def __init__(self):
        self.position = { axis: 0 for axis in config_base.keys() }
        self.backlash = { axis: -1 for axis in config_base.keys() }
        self.acceleration = { axis: -1 for axis in config_base.keys() }
        self.speed_max = { axis: -1 for axis in config_base.keys() }
        self.speed_monitor = { axis: -1 for axis in config_base.keys() }
        self.indexed_ids = { cfg['index']: axis for axis, cfg in config_base.items() }

        # Command channel to OpenOCD
        #self.telnet = telnetlib.Telnet("localhost", 4444)

    def global_reset(self):
        #self.telnet.write("reset\n")
        raise NotImplementedError

    def get_position(self, axis):
        return self.position[axis]

    def set_backlash(self, motor, value, override=False):
        if override or self.backlash[motor] != value:
            self.motor_command_word(motor, 'backlash', value)
            self.backlash[motor] = value

    def set_axes_enabled(self, enabled=True):
        self.global_command_byte('motor-status', enabled)

    def move_motor(self, name, kind, params):
        motor = name[5:]
        self.move_command(motor, kind, **params)

    def move_command(self, motor, kind, value=1, acceleration=100, max_speed=800, backlash=0):
        assert motor in config_base.keys()
        assert acceleration > 0
        assert max_speed > 0
        assert backlash >= 0

        self.set_acceleration(motor, acceleration)
        self.set_max_speed(motor, max_speed)
        self.set_backlash(motor, backlash)
        if kind == 'abs': self.move_absolute(motor, value)
        elif kind == 'rel': self.move_relative(motor, value)

    def stop(self, motor):
        assert motor in config_base.keys()
        self.motor_command_word(motor, 'steps', 0)

    def set_mode(self, motor, mode):
        assert mode.upper() in motor_modes.keys()
        assert motor in config_base.keys()
        self.motor_command_byte(motor, 'mode', motor_modes[mode.upper()])

    def set_hold_force(self, motor, hold):
        assert isinstance(hold, bool)
        self.motor_command_byte(motor, 'hold', hold)

    def set_direction_invert(self, motor, invert):
        assert isinstance(invert, bool)
        self.motor_command_byte(motor, 'dir-invert', invert)

    def set_acceleration(self, motor, acceleration, override=False):
        if override or self.acceleration[motor] != acceleration:
            self.motor_command_word(motor, 'acceleration', acceleration)
            self.acceleration[motor] = acceleration

    def set_max_speed(self, motor, max_speed, override=False):
        if override or self.speed_max[motor] != max_speed:
            self.motor_command_word(motor, 'speed-max', max_speed)
            self.speed_max[motor] = max_speed

    def fix_speed(self, motor, speed):
        self.motor_command_word(motor, 'speed-constant', speed)

    def move_relative(self, motor, displacement):
        return self._move(motor, self.position[motor] + displacement)

    def move_absolute(self, motor, target):
        return self._move(motor, target)

    def set_trigger_positions(self, positions, positions2=None):
        self.global_command_sequence('sequence-steps', positions)
        if positions2:
            self.global_command_sequence('sequence2-steps', positions2)

    def execute_trigger_positions(self, do=False):
        self.global_command_byte('sequence-cmd', do)

    def set_trigger_wait_time(self, shutter_time):
        self.global_command_byte('cam-shutter-time', shutter_time)

    def set_motion_wait_time(self, wait_time):
        self.global_command_word('cam-wait-time', wait_time, True)

    def _move(self, motor, target_position):
        at_target = (target_position == self.position[motor])
        if not at_target:
            self.motor_command_int(motor, 'position-target', target_position)

        return not at_target

    def set_origin(self, motor):
        assert motor in config_base.keys()
        self.position[motor] = 0
        self.motor_command_word(motor, 'position-current', 0)

    def set_led_state(self, lit=False):
        self.global_command_byte('led-state', lit)

    def set_led2_state(self, lit=False):
        self.global_command_byte('led2-state', lit)

    def con_plate_ana(self, cont=False):
        if cont:
            self.global_command_byte('plate-i-ana')
        else:
            self.global_command_byte('plate-o-ana')

    def con_plate_pol(self, cont=False):
        if cont:
            self.global_command_byte('plate-i-pol')
        else:
            self.global_command_byte('plate-o-pol')

    def set_led_fan_speed(self, speed=1.0):
        self.global_command_byte('led-fan-duty', int(255*speed))

    def set_cam_fan_speed(self, speed=1.0):
        self.global_command_byte('cam-fan-duty', int(255*speed))

    def register_address(self, motor, register):
        return motor_address[motor] + motor_register_offset[register]

    def motor_command_word(self):
        raise NotImplementedError

    def motor_command_int(self):
        raise NotImplementedError

    def motor_command_byte(self):
        raise NotImplementedError

    def global_command_byte(self):
        raise NotImplementedError

