# -*- coding: utf-8 -*-
# Importando librerias.
import itertools as it
import os.path as pth
import re
from fysom import Fysom
import numpy as np
from scipy import interpolate
# Importando archivos
from common import DebugLog, print_state_change
from motor_api import get_hardware_config

debug_log = DebugLog()

motor_indices = { motor: params['index'] for motor, params in get_hardware_config().items() }


class SweepController(object):
    # Order is [major axis, minor axis, fast axis]
    hardware_order = ['ana', 'y', 'x']
    filename_format = "{}/{:d}/{:02d}_{:02d}"
    filename_re = re.compile(r'(?P<kind>[^/]+)/(?P<ana>[0-9]+)/(?P<y>[0-9]+)_(?P<x>[0-9]+)')

    def __init__(self, cameras, motor_executor, notify_fun, debug=False):
        # Parseando los eventos que realiza el motor con los comandos que se le dan desde la web.
        self.cameras = cameras
        self.motor_exec = motor_executor
        self.notify_fun = notify_fun
        self.success = False
        events = [
            {'name': "start", 'src': "idle", 'dst': "preparing"},
            {'name': "motor-ready", 'src': "preparing", 'dst': "moving"},
            # Right after issuing commands for motor positioning and camera setup:
            {'name': "motor-ready", 'src': "moving", 'dst': "sweep-ready-mot"},
            {'name': "cam-ready", 'src': "moving", 'dst': "sweep-ready-cam"},
            # When both motors and camera are ready for a fast-axis sequence:
            {'name': "motor-ready", 'src': "sweep-ready-cam", 'dst': "sweeping"},
            {'name': "cam-ready", 'src': "sweep-ready-mot", 'dst': "sweeping"},
            # When a fast-axis sequence has completed, it's time for the next:
            {'name': "sweep-full", 'src': "sweeping", 'dst': "moving"},
            {'name': "stop", 'src': "*", 'dst': "idle"}
        ]
        callbacks = {
            'onleaveidle': self.reset,
            'onenterpreparing': self.prepare_motors,
            'onleavepreparing': self.preparation_done,
            'onentermoving': self.move,
            'onentersweeping': self.sweep_fast_axis,
            'onstop': self.halt
        }
        self.fsm = Fysom(initial='idle', events=events, callbacks=callbacks)
        self.stop = self.fsm.stop
        if debug:
            self.fsm.onchangestate = print_state_change

    def start(self, parameters):
        self.parameters = parameters
        cname = parameters['custom_name']
        angles_spec = parameters['limit_angle']
        dir_gen = ( (kind, idx_angle) for kind, angles in angles_spec.items() for idx_angle, angle in enumerate(angles) )
        dirs = [ pth.join(kind, str(idx_angle)) for kind, idx_angle in dir_gen ]
        self.cameras['cam0'].set_sweep_settings({'name': cname, 'directories': dirs, 'active': True})
        self.focus_reset()

    def reset(self, event):
        parameters = self.parameters
        self.motor_idle = True
        self.success = False
        self.max_speeds = parameters['speeds-max']
        self.accelerations = parameters['accelerations']
        self.exposure_time_ms = parameters['exposure-time']
        self.strides = parameters['strides']
        self.inverts = parameters['inverts']
        limit_row = parameters['limit_row']
        limit_column = parameters['limit_col']
        limit_angle = parameters['limit_angle']
        self.motion_wait_time_ms = parameters['wait']
        self.num_rows = limit_row + 1
        self.num_cols = limit_column + 1
        xpl_steps = len(limit_angle['xpl'])
        ppl_steps = len(limit_angle['ppl'])
        self.num_angles = xpl_steps + ppl_steps
        self.pictures_remaining = self.num_rows * self.num_cols * self.num_angles
        map_row = default_map_position(0, limit_row * self.strides['y'], 0, limit_row)
        map_col = default_map_position(0, limit_column * self.strides['x'], 0, limit_column)

        # Polarizer can stop twice at each step: one for XPLs and one for PPLs
        if isinstance(self.strides['pol'], list):
            half_travel = lut_map_position(self.strides['pol'])
        else:
            half_travel = default_map_position(0, self.strides['pol']*xpl_steps, 0, limit_angle)
        # Polarizer's angle 0 always marks the start of both XPL and PPL
        map_pol = half_travel[:xpl_steps][::-1] + half_travel[:ppl_steps]

        # Analyzer's map ignores the ana axis stride setting
        map_ana = limit_angle['xpl'][::-1] + limit_angle['ppl']

        self.map_position = {'x': map_col, 'y': map_row, 'ana': map_ana, 'pol': map_pol}

        # Angles position tags for filename generation
        logic_pos_angle = range(xpl_steps)[::-1] + range(ppl_steps)
        kind_pos_angle = ['xpl']*xpl_steps + ['ppl']*ppl_steps
        self.angle_pos_tag = zip(kind_pos_angle, logic_pos_angle)

        # Order is [major axis, minor axis, fast axis]
        self.sweep_order = ['y', 'x', 'ana']

        self.axes_stops = {'x': self.num_cols, 'y': self.num_rows, 'ana': self.num_angles}
        gen_sequence_args = [ self.axes_stops[ax] for ax in self.sweep_order ]
        self.sweep_commands = it.izip(*sweep_gen(*gen_sequence_args))

        self.last_major_tag = 0

        #todo Maybe receive the RBF spread
        self.focus_build_map()

        #metadata = {
        #    'nrow': self.num_rows,
        #    'ncol': self.num_cols,
        #    'pol-angle': self.num_angles,
        #    'strides': self.strides,
        #    'focus': {'table': list(self.focus_table),
        #              'spread': self.focus_spread},
        #}

        self.notify_fun("start")
        debug_log("FSM", "Preparing sweep...")

    def prepare_motors(self, event):
        # Parametros para la preparacion de los motores.
        for motor in motor_indices:
            self.motor_exec.set_acceleration(motor, self.accelerations[motor], True)
            self.motor_exec.set_max_speed(motor, self.max_speeds[motor], True)

        cam_trigger_time_ms = max(self.exposure_time_ms.values())
        self.motor_exec.set_trigger_wait_time(cam_trigger_time_ms)
        self.motor_exec.set_motion_wait_time(self.motion_wait_time_ms)
        will_move = False
        for motor in ('x', 'y', 'z'):
            if self.motor_exec.get_position(motor) != 0:
                debug_log("FSM", "Moving", motor.upper(), "axis to zero...")
                self.motor_exec.move_absolute(motor, 0)
                will_move = True
        if not will_move:
            debug_log("FSM", "All axes already at zero.")
            self.fsm.trigger("motor-ready")

    def preparation_done(self, event):
        # Esta funcion se ejecuta cuando la preparacion esta completa, con los axes y la posicion logica del lente.
        axes = self.strides.keys()
        self.position = { axis: 0 for axis in axes }
        self.logical_step = { axis: 0 for axis in axes }
        debug_log("FSM", "Motor preparation done. Starting sweep...")

    def halt(self, event):
        self.notify_fun("done")
        self.cameras['cam0'].set_sweep_settings({'active': False})
        self.motor_exec.execute_trigger_positions(False)
        if self.success:
            debug_log("FSM", "Stop sweep")
        else:
            debug_log("FSM", "Halt sweep")

    def sweep_fast_axis(self, event):
        # Movimientos de los axis hacia la posicion correcta de la imagen.
        debug_log("FSM", "sweep for pictures", self.logical_step)
        self.motor_exec.set_trigger_positions(self.seq_cols, self.seq_cols2)
        self.motor_exec.execute_trigger_positions(True)
        

    def move(self, event):
        # Movimientos del brazo del microscopio
        ax_major, ax_minor, ax_fast = self.sweep_order
        idx_major, idx_minor, seq_fastaxis = self.sweep_commands.next()
        if idx_major != self.last_major_tag:
            will_move = self.move_axis_mapped(ax_major, idx_major)
        else:
            will_move = self.move_axis_mapped(ax_minor, idx_minor)
        self.last_major_tag = idx_major
        n_rep = len(seq_fastaxis)
        major_gen = it.repeat(idx_major, n_rep)
        minor_gen = it.repeat(idx_minor, n_rep)
        canonical_gen = [major_gen, minor_gen, seq_fastaxis]
        indices = [self.sweep_order.index(axis) for axis in self.hardware_order]
        log_parts = zip(*[canonical_gen[ix] for ix in indices])
        phy_parts = [ list(self.angle_pos_tag[ang]) + [row, col] for ang, row, col in log_parts ]
        filenames = [ self.filename_format.format(kind, ang, row, col) for kind, ang, row, col in phy_parts ]
        exposures = [ self.exposure_time_ms[p[0]] for p in phy_parts ]
        self.cameras['cam0'].configure_fastaxis_scan({'seq': filenames, 'expo': exposures})

        self.position[ax_major] = self.map_position[ax_major][idx_major]
        self.position[ax_minor] = self.map_position[ax_minor][idx_minor]
        if ax_fast == 'ana':
            self.seq_cols = [self.map_position['ana'][ix] for ix in seq_fastaxis]
            self.seq_cols2 = [self.map_position['pol'][ix] for ix in seq_fastaxis]
        else:
            self.seq_cols = [self.map_position[ax_fast][ix] for ix in seq_fastaxis]
            self.seq_cols2 = None
        self.logical_step[ax_major] = idx_major
        self.logical_step[ax_minor] = idx_minor
        self.logical_step[ax_fast] = 0
        self.logical_step['pol'] = self.logical_step['ana']

        if not will_move:
            self.fsm.trigger("motor-ready")
            print ("light on")
            

    def move_axis_mapped(self, axis, index):
        will_move = False
        if axis in ('ana', 'pol'):
            will_move |= self.move_map('ana', index)
            will_move |= self.move_map('pol', index)
        else:
            will_move |= self.move_map(axis, index)

        return will_move

    def move_map(self, axis, index):
        pos = self.map_position[axis][index]
        will_move = self.motor_exec.move_absolute(axis, pos)
        if will_move:
            debug_log("FSM", "To move on axis", axis, "to pos:", pos)
        else:
            debug_log("FSM", "Axis", axis, "already at pos:", pos)

        return will_move

    def camera_event_dispatch(self, data):
        debug_log("->", "cam", data)
        if self.fsm.isstate('idle'):
            key = 'sweep'
            if key in data and data[key]:
                self.fsm.trigger("start")
        else:
            keys = ('saving', 'error-save')
            if any(k in data for k in keys):
                key = keys[0] if keys[0] in data else keys[1]
                file_saved = data[key]
                s_parts = self.filename_re.search(file_saved).groupdict()
                for ax in self.hardware_order:
                    s_parts[ax] = int(s_parts[ax])

            if keys[0] in data:
                ax_major, ax_minor, ax_fast = self.sweep_order
                self.position[ax_fast] = s_parts[ax_fast]
                self.notify_fun([s_parts[ax] for ax in ['kind', 'ana', 'y', 'x']])
                self.pictures_remaining -= 1
                if self.pictures_remaining > 0:
                    ax_fast = self.sweep_order[-1]
                    self.logical_step[ax_fast] += 1
                    if self.logical_step[ax_fast] == self.axes_stops[ax_fast]:
                        self.fsm.trigger("sweep-full")
                else:
                    self.success = True
                    self.fsm.trigger("stop")
            if keys[1] in data:
                debug_log("CAM", "error saving image at kind:", s_parts['kind'], "angle:", s_parts['ana'], "row:", s_parts['y'], "col:", s_parts['x'])
                self.fsm.trigger("stop")
            key = 'ready'
            if key in data and data[key]:
                self.fsm.trigger("cam-ready")

    def motor_done_notify(self):
        self.motor_idle = True
        if self.fsm.current in ('moving', 'preparing', 'sweep-ready-cam'):
            self.fsm.trigger("motor-ready")
        elif self.fsm.current == "sweeping":
            ax_fast = self.sweep_order[-1]
            debug_log("MOT", "in sweep, step:", self.logical_step[ax_fast])

    def focus_reset(self):
        # reset del foco de la camara
        self.focus_table = set()

    def focus_set(self):
        # Setting el foco de la camara
        point = (self.motor_exec.get_position('x'), self.motor_exec.get_position('y'), self.motor_exec.get_position('z'))
        self.focus_table.add(point)
        debug_log("FCS", "new point:", point)

    def focus_build_map(self, spread=None):
        if spread is None:
            col_width = self.num_cols * self.strides['x']
            row_width = self.num_rows * self.strides['y']
            spread = float(min(col_width, row_width))/6
        self.focus_spread = spread
        debug_log("FCS", "spread set to", self.focus_spread)

        if len(self.focus_table) > 0:
            x, y, z = zip(*self.focus_table)
            self.focus_map = interpolate.Rbf(x, y, z, epsilon=spread, function='gaussian')
            self.do_focus = True
        else:
            self.focus_map = lambda x, y: 0
            self.do_focus = False


def sweep_gen(steps_axis_major, steps_axis_minor, steps_axis_fast):
    # Assumption: A sweep always starts at physical coord (0, 0)
    dirs_axis_minor, dirs_axis_fast = gen_direction_steps(steps_axis_major, steps_axis_minor, [1, 1])
    indices_axis_minor_base = np.r_[0:steps_axis_minor]
    indices_axis_fast_base = np.r_[0:steps_axis_fast]
    indices_axis_minor = indices_axis_minor_base[::dirs_axis_minor[0]]
    if steps_axis_major > 1:
        cycles_axis_minor = ( indices_axis_minor_base[::d] for d in dirs_axis_minor[1:] )
        indices_axis_minor = np.hstack((indices_axis_minor, np.hstack(cycles_axis_minor)))
    dirs_axis_fast = (-1)**np.r_[0:indices_axis_minor.shape[0]] * dirs_axis_fast[0]
    axis_fast_sequences = [ indices_axis_fast_base[::d] for d in dirs_axis_fast ]
    pre = np.zeros((steps_axis_major * steps_axis_minor,), dtype=int)
    pre[::steps_axis_minor] = 1
    indices_axis_major = pre.cumsum() - 1

    return indices_axis_major, indices_axis_minor, axis_fast_sequences


def gen_direction_steps(number_of_rows, number_of_columns, principal_directions):
    steps_rows = (-1)**(np.r_[0:number_of_rows]) * principal_directions[0]
    steps_cols = (-1)**(np.r_[0:number_of_columns]) * principal_directions[1]

    return steps_rows, steps_cols


def default_map_position(abs_start, abs_end, log_start, log_end):
    """
    def line(x):
        assert log_start <= x <= log_end, "bad values: {} <? {} <? {}, (as={}, ae={})".format(log_start, x, log_end, abs_start, abs_end)
        #factor = float(abs_end - abs_start)/(log_end - log_start)
        #return int(round(abs_start + (x - log_start)*factor))
    return line
    """
    map_out = [ abs_start + ((x - log_start)*(abs_end - abs_start))/(log_end - log_start) for x in range(log_start, log_end+1) ]
    return map_out


def lut_map_position(lut):
    """
    def line(x):
        assert 0 <= x < len(lut)
        return lut[x]
    return line
    """

    return lut
