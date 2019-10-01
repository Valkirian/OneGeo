var UXLamp2 = new function() {

    var self = this;

    // Constants

    // Internal variables
    self.fan_speed = 100;

    self.create_widgets = function() {

        var dom_lamp2 = $('#lamp2');

        self.add_power_button(dom_lamp2);



    };


    self.add_power_button = function(dom_lamp2) {

        var callback = function(state) {
                console.log("Lamp2 new state: " + state);
                self.set_power(state);
            },
            button = Utils.create_switch_button(callback);

        dom_lamp2.append(button);
    };

    self.set_power = function(powered) {
        Comm.send_command('lamp2', 'power', { 'on': powered });
    };

    self._move = function(axis, kind, value, queue) {

        var command = 'move-' + kind,
            settings = {
                'acceleration': self.accelerations[axis],
                'max_speed': self.fan_speed,
                'backlash': self.backlash[axis]
            },
            queue = queue || false;

        if (queue || self.are_all_idle()) {
            settings['value'] = value;
            Comm.send_command(self.motor_name(axis), command, settings);
            self.all_idle = false;
            return true;
        } else {
            return false;
        }
    };

    self.move_relative = function(axis, steps, queue) {
        return self._move(axis, 'rel', steps, queue);
    };

    self.move_absolute = function(axis, position, queue) {
        return self._move(axis, 'abs', position, queue);
    };

    self._absolute_dir = function(axis, direction) {
        var invert = self.inverts[axis] ? -1 : 1,
            factor = direction === 1 ? invert : -1 * invert;

        return factor;
    };

    self.move_stride = function(axis, direction, queue) {

        var factor = self._absolute_dir(axis, direction);

        queue = queue || false;

        if (!$.isArray(self.strides[axis])) {
            console.log("st: " + self.strides[axis]);
            if (self.move_relative(axis, factor * self.strides[axis], queue)) {
                if (self.stride_callback) {
                    self.stride_callback(axis, direction);
                }
            }
        }
    };

    self.parse_stride_field = function(axis, source_string) {
        var new_vals = Utils.parse_string_list(source_string),
            str_value;

        if (new_vals.length == 1) {
            self.strides[axis] = new_vals[0];
            str_value = self.strides[axis];
        } else {
            self.strides[axis] = new_vals;
            str_value = new_vals.join(", ");
        }

        return str_value;
    };

    self.set_all_idle = function() {
        self.all_idle = true;
    };

    self.are_all_idle = function() {
        return self.all_idle;
    };

    self.motor_name = function(axis) {
        return "motor" + axis;
    };

};