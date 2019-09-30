var UXCam = new function() {

    var self = this;

    self.image_data_format = "png";

    self.exposure_manual = {}
    self.dom_cam = { 'cam0': $('#cam0 .input-group') };
    self.attr_currently_editing = '';
    self.fan_speed = 100;


    self.switch_exposure = function(camera) {

        if (!self.exposure_manual.hasOwnProperty(camera)) {
            self.exposure_manual[camera] = false;
        }

        self.exposure_manual[camera] ^= true;
        if (self.exposure_manual[camera]) {
            self.set_exposure_manual(camera);
        } else {
            self.set_exposure_auto(camera);
        }
    };

    self.set_exposure_manual = function(camera) {

        var selector = '#' + camera + ' ',
            shutter_speed = +$(selector + '#shutter_speed ~ input').val(),
            gain_analog = +$(selector + '#gain_analog ~ input').val(),
            button = $(selector + '#cam-mode');

        self.exposure_manual[camera] = true;
        if (shutter_speed != '') {
            //////////////////////////////////////////////////////////////////////////////////
            Comm.send_command(camera, 'set_cam_params', { 'shutter_speed': 10 });
            Comm.send_command(camera, 'set_cam_params', { 'awb_gains': [746, 665] });
            Comm.send_command(camera, 'set_cam_params', { 'gain_analog': 0 });
            //////////////////////////////////////////////////////////////////////////////////
            button.attr('class', 'btn btn-info');
            button.text('Exposure MANUAL');
        }


    };

    self.set_exposure_auto = function(camera) {

        var button = $('#' + camera + ' #cam-mode');

        self.exposure_manual[camera] = false;
        Comm.send_command(camera, 'set_cam_params', { 'shutter_speed': -1 });
        Comm.send_command(camera, 'set_cam_params', { 'gain_analog': -1 });

        button.attr('class', 'btn btn-warning');
        button.text('Exposure AUTO');
    };

    self.take_picture = function(camera) {
        Comm.send_command(camera, 'take_picture', {});
        //TODO have the picture data returned
        //     into a file, so that the user can download it
    };

    // Configure camera parameter controls for requesting parameter value
    // change to the server on Enter keypress after value change.
    self.set_widgets_behavior = function(camera) {

        self.dom_cam[camera].each(function(idx) {
            var me = $(this),
                o_span = me.find('span:first'),
                o_input = me.find('input'),
                attribute = o_span.attr('id'),
                setting = {},
                value;

            o_input.keypress(function(event) {
                // on Enter key
                if (event.keyCode == 13) {
                    console.log('Will send ' + o_input.val() + ' to prop ' + attribute);
                    if (attribute == 'awb_gains') {
                        value = Utils.parse_string_list(o_input.val());
                    } else {
                        value = +o_input.val();
                    }

                    setting[attribute] = value;
                    Comm.send_command(camera, 'set_cam_params', setting);
                }
            });

            o_input.focusin(function() {
                UXKeyboard.actions_enabled(false);
                self.attr_currently_editing = attribute;
            });
            o_input.focusout(function() {
                UXKeyboard.actions_enabled(true);
                self.attr_currently_editing = '';
            });

        });

        $('#' + camera + ' #cam-mode').click(function() {
            self.switch_exposure(camera);
        });


        self.add_fan_control(camera, $('#' + camera));
        self.add_power_button(camera, $('#' + camera));
    };

    self.set_power = function(camera, powered) {
        Comm.send_command(camera, 'power', { 'on': powered });
    };

    self.add_power_button = function(camera, dom_cam) {

        var callback = function(state) {
                console.log("cam new state: " + state);
                self.set_power(camera, state);
            },
            button = Utils.create_switch_button(callback, "power");

        dom_cam.prepend(button);
    };

    self.add_fan_control = function(camera, dom_cam) {

        var fan = Utils.build_input_group('fan', "Fan Speed", 'sublabel', {
                'complement-info': "%",
                'complement-title': "Speed",
                'input-type': 'range'
            }),
            speed_field = fan[1],
            cookie_speed = camera + '-fan-speed';

        // Persistence of speed field
        if (Cookies.get(cookie_speed) == null) {
            self.fan_speed = 100;
        } else {
            self.fan_speed = +Cookies.get(cookie_speed)
        }
        speed_field.val(self.fan_speed);

        // Behavior of axis speed field
        speed_field.change(function(event) {
            self.fan_speed = +speed_field.val();
            Cookies.set(cookie_speed, self.fan_speed);
            Comm.send_command(camera, 'fan-set', { 'speed': self.fan_speed / 100 });
        });

        dom_cam.prepend(fan);
    };

    self.update_configuration = function(camera, data) {

        //Sometimes data is already an object, sometimes it needs JSON.parse
        $.each(data, function(attribute, value) {
            if (attribute != "power") {
                var o_input = self.dom_cam[camera].find('#' + attribute + ' ~ input');
            }

            if (attribute != self.attr_currently_editing) {
                switch (attribute) {
                    case "awb_gains":
                        var gain_bl = value[0],
                            gain_rd = value[1],
                            outstr = gain_bl + ',' + gain_rd;
                        o_input.val(outstr);
                        break;

                    case "power":
                        var button = $('#' + camera + ' #power');
                        // Correlated to "state_table" in Utils.create_switch_button
                        if (value) {
                            button.removeClass('btn-success');
                            button.addClass('btn-danger');
                            button.text('ON');
                        } else {
                            button.removeClass('btn-danger');
                            button.addClass('btn-success');
                            button.text('OFF');
                        }
                        break;

                    default:
                        o_input.val(value);
                        break;
                }
            }
        });
    };
};