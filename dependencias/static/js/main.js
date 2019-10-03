$(document).ready( () => {

    var wsUri = "ws://" + window.location.host + "/websocket",
        sweep_is_on = false,
        sweep_timer,
        dom_sweep_state = $('#sweep #state'),
        dom_sweep_name = $('#sweep #name ~ input'),
        dom_sweep_limit_row = $('#sweep #row-stop ~ input'),
        dom_sweep_limit_col = $('#sweep #col-stop ~ input'),
        dom_sweep_steps_ppl = $('#sweep #ang-stop-ppl ~ input'),
        dom_sweep_steps_xpl = $('#sweep #ang-stop-xpl ~ input'),
        dom_sweep_row = $('#sweep #row ~ input'),
        dom_sweep_ang = $('#sweep #angle ~ input'),
        dom_sweep_col = $('#sweep #col ~ input'),
        dom_sweep_wait = $('#sweep #wait ~ input'),
        dom_sweep_row_triggers = $('#sweep #row-triggers ~ input'),
        dom_sweep_row_triggers_btn = $('#sweep #row-triggers ~ span button'),
        dom_sweep_focus_fields = $('#sweep #focus ~ input'),
        dom_sweep_delta_fields = $('#sweep #delta-pix ~ input'),
        dom_exposure_xpl = $('#sweep #expo-xpl'),
        dom_exposure_ppl = $('#sweep #expo-ppl'),
        S_S = 10,
        Flag_on = true,
        desensitizable_ctls = [
            dom_sweep_row, dom_sweep_col, dom_sweep_ang,
            dom_sweep_limit_row, dom_sweep_limit_col,
            dom_sweep_steps_ppl, dom_sweep_steps_xpl,
            dom_sweep_row_triggers, dom_sweep_name,
            dom_exposure_xpl, dom_exposure_ppl, dom_sweep_wait
        ];

    Comm.on_open = function(ev) {
        Utils.notification_event('success', 'Connection Established', 2500);
        UXLamp.set_power(false);
        UXLamp2.set_power(false);
    }

    Comm.on_error = function(ev) {
        Utils.notification_event('danger', "Error Occurred - " + ev.data, 1500);
    };

    Comm.on_close = function(ev) {
        Utils.notification_event('warning', "Connection Closed", 3500);
    };

    Comm.on_message = function(ev) {

        switch (jQuery.type(ev.data)) {
            case "object":

                var data_in = new Uint8Array(ev.data),
                    file_name = dom_sweep_name.val() + "." + UXCam.image_data_format,
                    blob = new Blob([data_in], { type: 'application/octet-binary' });
                console.log("Image size " + data_in.length);
                saveAs(blob, file_name);
                break;

            case "string":

                var msg = JSON.parse(ev.data),
                    type = msg.type,
                    data = msg.data,
                    name = msg.name,
                    utc_timestamp_ms = msg.time;

                switch (type) {
                    case 'cfg':
                        if (/cam/i.test(name)) {
                            UXCam.update_configuration(name, data);
                        } else if (name == "motor") {
                            UXMotors.create_widgets(data);
                        }
                        break;

                    case 'stt':

                        if (/cam/i.test(name)) {
                            if (data.hasOwnProperty('picture')) {
                                if (data['picture'] == "taken") {
                                    Utils.notification_event('success', 'Picture saved', 1500);
                                }
                            }
                        } else if (name == "motor") {
                            if (data.hasOwnProperty('move')) {
                                if (data['move']) {
                                    UXMotors.set_all_idle();
                                }
                            } else if (data.hasOwnProperty('position')) {
                                $.each(data['position'], function(axis, value) {
                                    UXMotors.update_position(axis, 'abs', value);
                                });
                            }
                        } else if (name == "sweep") {
                            console.info(data);
                            if (data.hasOwnProperty('complete')) {
                                sweep_is_on = !data['complete'];
                                ux_sweep_view(!data['complete']);
                            } else {
                                dom_sweep_ang.val(data['angle']);
                                dom_sweep_row.val(data['row']);
                                dom_sweep_col.val(data['col']);
                            }
                        } else {
                            console.log("stt:");
                        }
                        break;

                    case 'clc':
                        if (name == "delta-pix") {
                            $(dom_sweep_delta_fields[0]).val(data['deltas'][0]);
                            $(dom_sweep_delta_fields[1]).val(data['deltas'][1]);
                            console.info(data);
                        }
                        break;

                    case 'img':
                        if (name == "cam0") {
                            Utils.download_url(data['url'],
                                dom_sweep_name.val() + "." + UXCam.image_data_format);
                        }
                        break;

                    case 'fcs':
                        focus_value_0 = focus_value_1;
                        focus_value_1 = data;
                        break;

                    case 'sat':
                        sat_value = data;
                        break;


                    default:
                        console.info(data);
                        break;
                }
                break;
        }
    }
    var focus_value_0 = '',
        focus_value_1 = '',
        diff = 0;
    ////////////////////////////////////////////////////////////////////////////
    // Key assignments
    ////////////////////////////////////////////////////////////////////////////

    function sweep_state_control_wrap(callback_to_wrap) {
        var wrapped = function() {
            if (!sweep_is_on) {
                callback_to_wrap();
            }
        };
        return wrapped;
    }

    // 'm' key
    UXKeyboard.register_keypress(109, sweep_state_control_wrap(function() {
        UXCam.switch_exposure('cam0');
    }));

    // space bar
    UXKeyboard.register_keypress(32, sweep_state_control_wrap(function() {
        var filename = dom_sweep_name.val();
        UXCam.take_picture('cam0', filename);
    }));

    // 'a' key (97) and numpad 4 (52)
    UXKeyboard.register_keypress([52, 97], sweep_state_control_wrap(function() {
        UXMotors.move_stride('x', 0);
    }));

    // 'd' key (100) and numpad 6 (54)
    UXKeyboard.register_keypress([54, 100], sweep_state_control_wrap(function() {
        UXMotors.move_stride('x', 1);
    }));

    // 's' key (115) and numpad 5 (53)
    UXKeyboard.register_keypress([53, 115], sweep_state_control_wrap(function() {
        UXMotors.move_stride('y', 0);
    }));

    // 'w' key (119) and numpad 8 (56)
    UXKeyboard.register_keypress([56, 119], sweep_state_control_wrap(function() {
        UXMotors.move_stride('y', 1);
    }));

    // 'q' key (113) and  numpad 7 (55)
    UXKeyboard.register_keypress([55, 113], sweep_state_control_wrap(function() {
        UXMotors.move_stride('z', 1);
    }));

    // 'e' key (101) and numpad 9 (57)
    UXKeyboard.register_keypress([57, 101], sweep_state_control_wrap(function() {
        UXMotors.move_stride('z', 0);
    }));

    // 'z' key (122) and numpad 1 (49)
    UXKeyboard.register_keypress([122, 49], sweep_state_control_wrap(function() {
        //UXMotors.move_stride('pol', 0);
        //UXMotors.move_stride('ana', 0, true);        
        //UXMotors.move_relative('pol', 2);
        S_S = S_S - 0.5;
        Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': S_S });
        Utils.notification_event('info', "Shutter Speed set at: " + S_S.toString());
    }));

    // 'x' key (120) and numpad 3 (51)
    UXKeyboard.register_keypress([120, 51], sweep_state_control_wrap(function() {
        //UXMotors.move_stride('pol', 1);
        //UXMotors.move_stride('ana', 1, true);
        //UXMotors.move_relative('ana', 2);
        S_S = S_S + 0.5;
        Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': S_S });
        Utils.notification_event('info', "Shutter Speed set at: " + S_S.toString());
    }));

    UXKeyboard.register_keypress([86], sweep_state_control_wrap(function() {
        S_S = S_S + 0.5;
        Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': S_S });
        Utils.notification_event('info', "Shutter Speed set at: " + S_S.toString());
    }));
    UXKeyboard.register_keypress([67], sweep_state_control_wrap(function() {
        S_S = S_S - 0.5;
        Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': S_S });
        Utils.notification_event('info', "Shutter Speed set at: " + S_S.toString());
    }));

    UXMotors.stride_callback = function(axis, direction) {

        if (direction === 1) {
            switch (axis) {
                case 'x':
                    dom_sweep_col.val(+dom_sweep_col.val() - 1);
                    break;

                case 'y':
                    dom_sweep_row.val(+dom_sweep_row.val() - 1);
                    break;

                case 'ana':
                    dom_sweep_ang.val(+dom_sweep_ang.val() - 1);
                    break;
            }
        } else {
            switch (axis) {
                case 'x':
                    dom_sweep_col.val(+dom_sweep_col.val() + 1);
                    break;

                case 'y':
                    dom_sweep_row.val(+dom_sweep_row.val() + 1);
                    break;

                case 'ana':
                    dom_sweep_ang.val(+dom_sweep_ang.val() + 1);
                    break;
            }
        }
    };

    dom_sweep_state.click(function() {
        sweep_is_on ^= true;
        sweep_is_on = ux_sweep_set(sweep_is_on);
        ux_sweep_view(sweep_is_on);
    });

    function ux_sweep_view(state) {
        if (state) {
            dom_sweep_state.attr('class', 'btn btn-danger');
            dom_sweep_state.text('Sweep ON');
            sweep_timer = Date.now();
            msg = "Sweep started.";
            Utils.notification_event('info', msg, 2000);
        } else {
            dom_sweep_state.attr('class', 'btn btn-success');
            dom_sweep_state.text('Sweep OFF');
            var dt = Date.now() - sweep_timer,
                minutes = Math.floor(dt / 60e3),
                seconds = Math.round(dt / 1e3) - 60 * minutes,
                msg = "Sweep done in " + minutes + " min " + seconds + " seg.";
            Utils.notification_event('info', msg);
        }
    }

    function ux_sweep_set(state) {
        if (state) {
            var cam = "cam0",
                limit_stack = [],
                ppl_steps = Utils.parse_string_list($(dom_sweep_steps_ppl.get(0)).val()),
                xpl_steps = Utils.parse_string_list($(dom_sweep_steps_xpl.get(0)).val()),
                ppl_step_amount_str = $(dom_sweep_steps_ppl.get(1)).val(),
                xpl_step_amount_str = $(dom_sweep_steps_xpl.get(1)).val(),
                ppl_step_amount, xpl_step_amount, exposure_times,
                num_angles, limit_angle, success;

            if (ppl_step_amount_str == '') {
                ppl_step_amount = ppl_steps.length;
                $(dom_sweep_steps_ppl.get(1)).val(ppl_step_amount);
            } else ppl_step_amount = +ppl_step_amount_str;
            if (ppl_steps.length < ppl_step_amount) {
                Utils.notification_event('danger', "Specified PPL steps more than are specified.");
                return false;
            }

            if (xpl_step_amount_str == '') {
                xpl_step_amount = xpl_steps.length;
                $(dom_sweep_steps_xpl.get(1)).val(xpl_step_amount);
            } else xpl_step_amount = +xpl_step_amount_str;
            if (xpl_steps.length < xpl_step_amount) {
                Utils.notification_event('danger', "Specified XPL steps more than are specified.");
                return false;
            }

            // Check there are enough Polarizer steps in its stride to match those
            // specified for the Analyzer
            num_angles = Math.max(ppl_step_amount, xpl_step_amount);
            success = parse_stride_for_sweep('pol', num_angles);
            if (!success) return;
            limit_angle = {
                'ppl': ppl_steps.slice(0, ppl_step_amount),
                'xpl': xpl_steps.slice(0, xpl_step_amount)
            }

            dom_sweep_focus_fields.each(function() {
                limit_stack.push(+$(this).val());
            });

            if (!UXCam.exposure_manual[cam]) {
                UXCam.set_exposure_manual(cam);
            }

            exposure_times = {
                'xpl': +dom_exposure_xpl.val(),
                'ppl': +dom_exposure_ppl.val()
            }

            setTimeout(function() {
                Comm.send_command('sweep', 'start', {
                    'custom_name': dom_sweep_name.val(),
                    'speeds-max': UXMotors.speeds,
                    'exposure-time': exposure_times,
                    'wait': +dom_sweep_wait.val(),
                    'accelerations': UXMotors.accelerations,
                    'strides': UXMotors.strides,
                    'inverts': UXMotors.inverts,
                    'backlash': UXMotors.backlash,
                    'limit_angle': limit_angle,
                    'limit_row': +dom_sweep_limit_row.val(),
                    'limit_col': +dom_sweep_limit_col.val(),
                    'spec_stack': limit_stack
                });
            }, 1000);

            return true;
        } else {
            Comm.send_command('sweep', 'stop', {});
            return false;
        }
    }

    function parse_stride_for_sweep(axis, stride_amount) {
        if ($.isArray(UXMotors.strides[axis])) {
            if (UXMotors.strides[axis].length >= stride_amount) {
                var strides = [];
                $.each(UXMotors.strides[axis], function(index, value) {
                    strides.push(value);
                });
                UXMotors.strides[axis] = strides;
            } else {
                var axis_name = $('#motor #' + axis + ' .panel-heading').html(),
                    msg = "Number of steps for axis " + axis_name + " must be at least " + stride_amount;
                Utils.notification_event('danger', msg);

                return false;
            }
        }

        return true;
    }

    dom_sweep_row_triggers.keypress(function(event) {
        if (event.which == 13)
            triggers_parse_fun();
    });
    dom_sweep_row_triggers_btn.click(function(event) {
        triggers_parse_fun();
    });

    function triggers_parse_fun() {
        //TODO both the UC and the camera driver receive times in millisecond units,
        //maybe the 1000 factor is unnecessary
        var sequence = Utils.parse_string_list(dom_sweep_row_triggers.val()),
            wait_time = 1000 * (+dom_exposure_time.val());

        Comm.send_command('events', "wait-time", { 'time': wait_time });
        Comm.send_command('events', "trigger-sequence", { 'sequence': sequence });
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    $(' #auto-focus').click(function() {
        //Utils.notification_event('info', "Autofocus started");
        Utils.notification_event('info', wsUri);

        //Utils.notification_event('info', "Focus: " + focus_value_0.toString());
        //if (focus_value_1 != '' && focus_value_0 != '') {
        //  diff = (focus_value_1 - focus_value_0) / focus_value_1;
        // var dir = 1;
        // while (Math.abs(diff) >= 0.02) {
        //   if (focus_value_1 < focus_value_0) {
        //     dir = dir * -1;
        //           }
        //         diff = setTimeout(TOcall(dir), 1500);
        //   }

        //        function TOcall(dir) {
        //          UXMotors.move_relative('z', 2 * dir);
        //        diff = (focus_value_1 - focus_value_0) / focus_value_1;
        //      return diff
        //     }
        //  }
    });


    $(' #auto-sat').click(function() {
        Utils.notification_event('info', "Autoexpossure started");
        shutter_speed = 0;
        while (ss <= 250) {
            shutter_speed = setTimeout(send_ss(shutter_speed), 1000);
        }

        function send_ss() {
            shutter_speed = shutter_speed + 0.1;
            Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': shutter_speed });
            return shutter_speed
        }
        Utils.notification_event('info', "Shutter Speed set at:" + shutter_speed.toString());
    });

    $(' #ana-in').click(function() {
        $('#ana').attr('class', 'btn btn-warning');
        Comm.send_command('plate', 'ana', { 'on': true })
        setTimeout(function() { $('#ana').attr('class', 'btn') }, 2000);
    })
    $(' #ana-out').click(function() {
        $('#ana').attr('class', 'btn btn-warning');
        Comm.send_command('plate', 'ana', { 'on': false })
        setTimeout(function() { $('#ana').attr('class', 'btn') }, 2000);
    })

    $(' #pol-in').click(function() {
        $('#pol').attr('class', 'btn btn-warning');
        Comm.send_command('plate', 'pol', { 'on': true })
        setTimeout(function() { $('#pol').attr('class', 'btn') }, 2000);
    })
    $(' #pol-out').click(function() {
        $('#pol').attr('class', 'btn btn-warning');
        Comm.send_command('plate', 'pol', { 'on': false })
        setTimeout(function() { $('#pol').attr('class', 'btn') }, 2000);
    })

    $(' #Shutdown').click(function() {


        if (Flag_on) {
            $('#Shutdown').attr('class', 'btn btn-warning');
            Origin_fun();
            xpl_Or(7000);

            setTimeout(function() {
                Comm.send_command('lamp', 'power', { 'on': false });
                Comm.send_command('cam0', 'power', { 'on': false })
            }, 500);
            setTimeout(function() {
                Comm.send_command('lamp', 'fan-set', { 'speed': 0 });
                Comm.send_command('cam0', "fan-set", { 'speed': 0 })
            }, 3500);
            setTimeout(function() {
                $('#Shutdown').text('Start');
                Utils.notification_event('danger', "start")
            }, 10000);
            setTimeout(function() { $('#Shutdown').attr('class', 'btn btn-success') }, 10000);
            setTimeout(function() { Comm.send_command('motor', 'power', { 'level': false }) }, 11000);
            setTimeout(function() { Utils.notification_event('danger', "Please kill tmux process on a terminal : tmux kill-session -t capture") }, 7000);
        } else {
            $('#Shutdown').attr('class', 'btn btn-warning');
            On_start();
            setTimeout(function() { $('#Shutdown').text('Shutdown') }, 2000);
            setTimeout(function() { $('#Shutdown').attr('class', 'btn btn-danger') }, 2000);
            setTimeout(function() { Comm.send_command('motor', 'power', { 'level': true }) }, 2100);
        }
    });


    function toOrigin(axis, val = 0) {
        if (UXMotors.get_position(axis) != val) {
            UXMotors.move_absolute(axis, val);
            return false
        } else {
            return true
        }
    }


    function Origin_fun() {
        var Flag = false,
            tm = 3500;
        Flag = toOrigin('x', 0);
        if (Flag) {
            tm = 500;
        }
        setTimeout(function() { toOrigin('y', 0) }, tm);
        $('#go-origin').attr('class', 'btn btn-warning');
        setTimeout(function() { $('#go-origin').attr('class', 'btn btn-success') }, 2000 + tm);
        setTimeout(function() { $('#go-origin').attr('class', 'btn') }, 6000 + tm);
        setTimeout(function() { Utils.notification_event('info', "Origin position") }, 2000 + tm);
    }

    $('#go-origin').click(function() {
        Origin_fun()
    });

    function xpl_Or(tmd = 0) {
        var Flag = false,
            tm = 3500 + tmd;
        Flag = toOrigin('pol', 0);
        if (Flag) {
            tm = 500 + tmd;
        }
        setTimeout(function() { toOrigin('ana', 0) }, tm);
        $('#go-xpl').attr('class', 'btn btn-warning');
        $('#go-ppl').attr('class', 'btn');

        setTimeout(function() { $('#go-xpl').attr('class', 'btn btn-success') }, 2000 + tm);
        setTimeout(function() { $('#go-xpl').attr('class', 'btn') }, 6000 + tm);
        S_S = Number(dom_exposure_xpl.val());

        Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': S_S });

        setTimeout(function() { Utils.notification_event('info', " XPL Orientation ; Shutter Speed set at:" + S_S) }, 2000 + tm);
    }

    $('#go-xpl').click(function() {
        xpl_Or();
    });







    $('#go-ppl').click(function() {
        var Flag = false,
            tm = 3500;
        Flag = toOrigin('pol', 0);
        if (Flag) {
            tm = 500;
        }
        setTimeout(function() { toOrigin('ana', 356) }, tm);
        $('#go-ppl').attr('class', 'btn btn-warning');
        $('#go-xpl').attr('class', 'btn');

        setTimeout(function() { $('#go-ppl').attr('class', 'btn btn-success') }, 2000 + tm);
        setTimeout(function() { $('#go-ppl').attr('class', 'btn') }, 6000 + tm);
        S_S = Number(dom_exposure_ppl.val());
        Comm.send_command('cam0', 'set_cam_params', { 'shutter_speed': S_S });

        setTimeout(function() { Utils.notification_event('info', " PPL Orientation ; Shutter Speed set at:" + S_S) }, 2000 + tm);
    });

    $('#go-center').click(function() {
        var Flag = false,
            tm = 3500;
        Flag = toOrigin('x', 1320);
        if (Flag) {
            tm = 500;
        }
        setTimeout(function() { toOrigin('y', 500) }, tm);
        $('#go-center').attr('class', 'btn btn-warning');

        setTimeout(function() { $('#go-center').attr('class', 'btn btn-success') }, 2000 + tm);
        setTimeout(function() { $('#go-center').attr('class', 'btn') }, 6000 + tm);

        setTimeout(function() { Utils.notification_event('info', "Center position") }, 2000 + tm);
    });

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Initialization actions
    $('#tabs').tab();
    UXKeyboard.actions_enabled(true);
    Utils.init_notification_box();
    Comm.start(wsUri);
    UXCam.set_widgets_behavior('cam0');
    UXLamp.create_widgets();
    UXLamp2.create_widgets();
    ux_sweep_view(sweep_is_on);
    Cookies.defaults['expires'] = 5000;
    $.each(desensitizable_ctls, function(index, ctl) {
        UXKeyboard.desensitize_in_field(ctl);
    });
    /////////////////////////////////////////////////////////////////////////////////////////////
    On_start();

    function On_start() {
        setTimeout(function() { UXCam.set_exposure_manual('cam0') }, 5000);
        setTimeout(function() { Comm.send_command('lamp', 'fan-set', { 'speed': 0.6 }) }, 3000);
        setTimeout(function() { Comm.send_command('cam0', "fan-set", { 'speed': 1 }) }, 3200);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////



    Utils.assign_cookie(dom_sweep_name, "sweep-name", "sweep");
    Utils.assign_cookie(dom_sweep_limit_row, "sweep-limit-row", 0);
    Utils.assign_cookie(dom_sweep_limit_col, "sweep-limit-col", 0);
    Utils.assign_cookie(dom_sweep_steps_ppl, "sweep-steps-ppl");
    Utils.assign_cookie(dom_sweep_steps_xpl, "sweep-steps-xpl");
    Utils.assign_cookie(dom_exposure_xpl, "sweep-exposure-xpl");
    Utils.assign_cookie(dom_exposure_ppl, "sweep-exposure-ppl");

});