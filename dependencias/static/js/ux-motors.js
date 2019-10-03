var UXMotors = new function() {

    var self = this;

    // Constants
    self.default_max_speed = 800;
    self.default_acceleration = 200;
    self.default_strides_per_step = 20;

    // Internal variables
    self.all_idle = true;
    self.strides = {};
    self.speeds = {};
    self.accelerations = {};
    self.inverts = {};
    self.position_fields = {};
    self.backlash = {};

    // Public variables
    self.stride_callback = undefined;
    self.hardware_settings = undefined;

    self.create_widgets = function(settings) {

        var dom_motors = $('#motor'),
            axis_order = Utils.sort_keys_by_value(settings,
                function(entry) {
                    return entry['index'];
                });
        self.add_power_button(dom_motors);
        self.add_reset_button(dom_motors);

        $.each(axis_order, function(index, axis) {
            var config = settings[axis],
                widgets = self.draw_motor_group(axis, config['name']);

            dom_motors.append(widgets['panel']);
            //self.populate_stepmodes(axis, widgets['stepmode'], config['stepmodes']);
            self.set_widgets_behavior(axis, widgets);
        });

        self.hardware_settings = settings;
    };

    self.draw_motor_group = function(axis, name) {

        var panel = $(document.createElement('div')),
            panel_head = $(document.createElement('div')),
            panel_body = $(document.createElement('div')),
            controls_steps, controls_speed, controls_acceleration, controls_stride,
            inversion_group, inversion_widget;
        //stepmode_group, stepmode_widgets;

        panel.attr('class', "panel panel-default");
        panel.attr('id', axis);
        panel.append(panel_head, panel_body);

        panel_head.attr('class', "panel-heading");
        panel_head.append(name);
        panel_body.attr('class', "panel-body");

        controls_position = self.build_position_group();
        controls_steps = Utils.build_input_group('steps', "Steps to go", 'button', {
            'complement-info': "Go!",
            'input-type': 'number'
        });
        controls_speed = Utils.build_input_group('speed', "Max Speed", 'sublabel', {
            'complement-info': "sps",
            'complement-title': "Steps per second",
            'input-type': 'number'
        });
        controls_acceleration = Utils.build_input_group('accel', "Acceleration", 'sublabel', {
            'complement-info': "sps²",
            'complement-title': "Steps per second squared",
            'input-type': 'number'
        });
        controls_stride = Utils.build_input_group('stride', "Stride", 'sublabel', {
            'complement-info': "steps",
            'complement-title': "Steps per stride (triggered by keyboard)"
        });
        controls_lash = Utils.build_input_group('lash', "Backlash", 'sublabel', {
            'complement-info': "steps",
            'complement-title': "Steps per direction change",
            'input-type': 'number'
        });

        inversion_group = Utils.create_input_group('invert', "Invert direction?");
        inversion_widget = Utils.create_checkbox();
        inversion_group.append(inversion_widget);

        panel_body.append(controls_position[0], controls_steps[0], controls_speed[0],
            controls_acceleration[0], controls_stride[0], controls_lash[0],
            //stepmode_group, inversion_group);
            inversion_group);

        return {
            'panel': panel,
            'fields': {
                'pos': controls_position.slice(1),
                'steps': controls_steps.slice(1),
                'speed': controls_speed[1],
                'accel': controls_acceleration[1],
                'lash': controls_lash[1],
                'stride': controls_stride[1]
            },
            'invert': inversion_widget
        };
        //'stepmode': stepmode_widgets};
    };

    self.set_widgets_behavior = function(axis, widgets) {

        var cookie_speed = 'mot-speed-' + axis,
            cookie_lash = 'mot-lash-' + axis,
            cookie_accel = 'mot-accel-' + axis,
            cookie_steps = 'mot-steps-' + axis,
            cookie_stride = 'mot-stride-' + axis,
            cookie_invert = 'mot-invert-' + axis,
            stride_field = widgets['fields']['stride'],
            speed_field = widgets['fields']['speed'],
            lash_field = widgets['fields']['lash'],
            accel_field = widgets['fields']['accel'],
            steps_field = widgets['fields']['steps'][0],
            steps_button = widgets['fields']['steps'][1],
            invert_box = widgets['invert'],
            position_field = widgets['fields']['pos'][0],
            position_button_reset = widgets['fields']['pos'][1],
            position_button_go = widgets['fields']['pos'][2];

        UXKeyboard.field_is_numeric(speed_field);
        UXKeyboard.field_is_numeric(lash_field);
        UXKeyboard.field_is_numeric(accel_field);
        UXKeyboard.desensitize_in_field(stride_field);
        UXKeyboard.field_is_numeric(steps_field);
        UXKeyboard.field_is_numeric(position_field);
        self.position_fields[axis] = position_field;

        // Persistence of axis steps field
        if (Cookies.get(cookie_steps) != null) {
            steps_field.val(+Cookies.get(cookie_steps));
        }

        // Behavior and persistence of axis steps field
        var steps_parse_fun = function() {
            var steps = +steps_field.val(),
                speed = +speed_field.val(),
                factor = self._absolute_dir(axis, Math.sign(steps));

            self.move_relative(axis, factor * Math.abs(steps), speed);
            Cookies.set(cookie_steps, steps);
        };
        steps_field.keypress(function(event) {
            if (event.which == 13) // on Enter key
                steps_parse_fun();
        });
        steps_button.click(function(event) {
            steps_parse_fun();
        });

        // Persistence of axis stride field
        var str_value;
        if (Cookies.get(cookie_stride) == null) {
            self.strides[axis] = self.default_strides_per_step;
            str_value = self.strides[axis];
        } else {
            str_value = self.parse_stride_field(axis, Cookies.get(cookie_stride));
        }
        stride_field.val(str_value);

        // Behavior of axis stride field
        stride_field.change(function(event) {
            Cookies.set(cookie_stride,
                self.parse_stride_field(axis, stride_field.val()));
        });

        // Persistence of axis speed field
        if (Cookies.get(cookie_speed) == null) {
            self.speeds[axis] = self.default_max_speed;
        } else {
            self.speeds[axis] = +Cookies.get(cookie_speed)
        }
        speed_field.val(self.speeds[axis]);

        // Behavior of axis speed field
        speed_field.change(function(event) {
            self.speeds[axis] = +speed_field.val();
            Cookies.set(cookie_speed, self.speeds[axis]);
            console.log(self.speeds);
        });

        // Persistence of axis acceleration field
        if (Cookies.get(cookie_accel) == null) {
            self.accelerations[axis] = self.default_acceleration;
        } else {
            self.accelerations[axis] = +Cookies.get(cookie_accel)
        }
        accel_field.val(self.accelerations[axis]);

        // Behavior of axis acceleration field
        accel_field.change(function(event) {
            self.accelerations[axis] = +accel_field.val();
            Cookies.set(cookie_accel, self.accelerations[axis]);
            console.log(self.accelerations);
        });

        // Persistence of axis backlash field
        if (Cookies.get(cookie_lash) == null) {
            self.backlash[axis] = 0;
        } else {
            self.backlash[axis] = +Cookies.get(cookie_lash)
        }
        lash_field.val(self.backlash[axis]);

        // Behavior of axis backlash field
        lash_field.change(function(event) {
            self.backlash[axis] = +lash_field.val();
            Cookies.set(cookie_lash, self.backlash[axis]);
        });

        // Persistence of axis invert property
        if (Cookies.get(cookie_invert) == null) {
            self.inverts[axis] = false;
        } else {
            self.inverts[axis] = JSON.parse(Cookies.get(cookie_invert));
        }
        invert_box.prop('checked', self.inverts[axis]);

        // Behavior of axis invert checkbox
        invert_box.change(function() {
            self.inverts[axis] = $(this).is(":checked");
            Cookies.set(cookie_invert, self.inverts[axis]);
            console.log("Invert of axis " + axis + " set to: " + self.inverts[axis]);
        });

        // Behavior of axis position field
        position_field.val(0);
        var position_parse_fun = function() {
            self.move_absolute(axis, +position_field.val());
        };
        position_field.keypress(function(event) {
            if (event.which == 13) // on Enter key
                position_parse_fun();
        });
        position_button_go.click(function(event) {
            position_parse_fun();
        });
        position_button_reset.click(function(event) {
            Comm.send_command(self.motor_name(axis), "axis-reset", {});
            position_field.val(0);
        });
    };

    self.build_position_group = function() {
        var base_group = Utils.build_input_group('position', "Position", 'button', {
                'complement-info': "Reset",
                'input-type': 'number'
            }),
            go_button = Utils.create_button();

        go_button.text("Go!");
        go_button.addClass("btn-warning");

        base_group[2].parent().append(go_button);
        base_group.push(go_button);

        return base_group;
    };

    self.update_position = function(axis, kind, value) {
        var field = self.position_fields[axis];

        if (kind == "abs") {
            field.val(value);
        } else if (kind == "rel") {
            field.val(+field.val() + value);
        }
    };
    self.get_position = function(axis) {
        var field = self.position_fields[axis];
        return field.val();

    };

    self.add_power_button = function(dom_motors) {

        var callback = function(state) {
                console.log("Motor new state: " + state);
                Comm.send_command('motor', 'power', { 'level': state });
            },
            button = Utils.create_switch_button(callback);

        dom_motors.append(button);
        Comm.send_command('motor', 'power', { 'level': false });
    };

    self.add_reset_button = function(dom_motors) {

        var button = Utils.create_button();

        button.addClass('btn-info');
        button.text('Reset Axes Cursors');

        dom_motors.append(button);

        //Behavior of reset button
        button.click(function() {
            Comm.send_command('motor', 'axes-reset', {});
            $.each(self.position_fields, function(axis, field) {
                field.val(0);
            });
        });
    };

    self.populate_stepmodes = function(axis, widgets, config) {

        var dropdown = widgets[1],
            dropdown_button = widgets[0],
            new_item_str, new_item,
            indices = Utils.sort_indices(config['indices']),
            i, idx, len = indices.length,
            cookie_stepmode = 'mot-stepmode-' + axis,
            set_step_mode;

        for (i = 0; i < len; ++i) {
            idx = indices[i];
            new_item_str = '<li><a href="#" id="' +
                config['modes'][idx] + '" data-value=' +
                config['indices'][idx] + '>' +
                config['modes'][idx] + '</a></li>';
            new_item = $(new_item_str);
            dropdown.append(new_item);
        }

        set_step_mode = function(index) {
            var motor_name = self.motor_name(axis),
                obj = dropdown.find("[data-value=" + index + "]"),
                mode_name = obj.html();

            console.log("motor on axis " + axis + " mode index: " + obj.attr('data-value'));
            dropdown_button.html(mode_name + ' <span class="caret"></span>');

            Comm.send_command(motor_name, 'modestep', mode_name);
        };

        dropdown.find('li a').click(function() {
            var obj = $(this),
                val = obj.attr('data-value');

            set_step_mode(val);
            Cookies.set(cookie_stepmode, val);
        });

        if (Cookies.get(cookie_stepmode) == null) {
            step_mode_index = 0;
        } else {
            step_mode_index = +Cookies.get(cookie_stepmode)
        }
        set_step_mode(step_mode_index);
    };

    self._move = function(axis, kind, value, queue) {

        var command = 'move-' + kind,
            settings = {
                'acceleration': self.accelerations[axis],
                'max_speed': self.speeds[axis],
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