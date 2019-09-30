var Utils = new function() {

    var self = this;

    self.dom_notify = $('#notification .alert');

    self.parse_string_list = function(list) {

        var assembly = [];

        if (list.trim() != '') {
            $.each(list.split(','), function(index, val) {
                assembly.push(+val.trim());
            });
        }

        return assembly;
    };


    self.fraction2float = function(fraction_str) {

        var parts = fraction_str.split('/'),
            num = +parts[0],
            den = +parts[1];

        return (num/den).toFixed(4);
    };


    self.frame_raw2src = function (buffer) {

        var binary = '',
            len = buffer.byteLength;

        for (var i = 0; i < len; i++) {
            binary += String.fromCharCode(buffer[i]);
        }

        return window.btoa(binary);
    };


    self.sort_indices = function (array) {

        var len = array.length,
            indices = new Array(len);

        for (var i = 0; i < len; ++i) indices[i] = i;
        indices.sort(function (a, b) { return array[a] < array[b] ? -1 : array[a] > array[b] ? 1 : 0; });

        return indices;
    };


    self.sort_keys_by_value = function(object, sort_value_getter) {

        var key_registry = {}, key_order = [], indices = [],
            min_index = -1, key_count = 0;

        $.each(object, function(key, entry) {
            var index = sort_value_getter(entry);

            key_registry[index] = key;
            indices.push(index);
            ++key_count;
            if ( (min_index == -1) || (index < min_index) ) {
                min_index = index;
            }
        });
        indices = indices.sort();
        
        for(var i = 0; i < key_count; ++i) {
            key_order.push(key_registry[indices[i]]);
        }

        return key_order;
    };


    self.create_input_group = function(id, label) {
        var dom_group = $(document.createElement('div')), 
            dom_label = $(document.createElement('span'));

        dom_group.attr('class', "input-group input-group-sm");
        dom_group.attr('id', id);
        dom_group.append(dom_label);

        dom_label.attr('class', "input-group-addon");
        dom_label.append(label);

        return dom_group;
    };


    self.build_input_group = function(id, label, complement, kwargs) {

        var args = kwargs || {},
            complement = complement || "none",
            dom_group = self.create_input_group(id, label), 
            dom_input = $(document.createElement('input')),
            elements = [dom_group, dom_input];

        dom_group.append(dom_input);
        dom_input.attr('class', "form-control");
        if (args.hasOwnProperty('input-type')) {
            dom_input.attr('type', args['input-type']);
        }
        if (args.hasOwnProperty('value')) {
            dom_input.attr('value', args['value']);
        }
        
        switch (complement) {
            case "none":
                break;

            case "button":
                var dom_button_cnt = $(document.createElement('span')),
                    dom_button = $(document.createElement('button'));
                dom_group.append(dom_button_cnt);
                elements.push(dom_button);
                dom_button_cnt.append(dom_button);
                dom_button_cnt.attr('class', "input-group-btn");
                dom_button.attr('class', "btn");
                dom_button.attr('type', "button");
                dom_button.append(args['complement-info']);
                if (args.hasOwnProperty('complement-subclass')) {
                    dom_button.addClass(args['complement-subclass']);
                }
                else {
                    dom_button.addClass("btn-default");
                }
                break;

            case "sublabel":
                var dom_sublabel = $(document.createElement('span'));
                dom_group.append(dom_sublabel);
                dom_sublabel.attr('class', "input-group-addon");
                dom_sublabel.append(args['complement-info']);
                if (args.hasOwnProperty('complement-title')) {
                    dom_sublabel.attr('title', args['complement-title']);
                }
                break;
        }

        return elements;
    };


    self.create_dropdown = function() {

        var dom_button = $(document.createElement('button')),
            dom_list = $(document.createElement('ul')),
            dom_caret = $(document.createElement('span'));

        dom_button.append(dom_caret);
        dom_button.attr('class', "btn btn-default dropdown-toggle");
        dom_button.attr('type', "button");
        dom_button.attr('data-toggle', "dropdown");
        dom_button.attr('aria-expanded', "false");
        dom_list.attr('class', "dropdown-menu");
        dom_list.attr('role', "menu");
        dom_caret.attr('class', "caret");

        return [dom_button, dom_list];
    };


    self.create_checkbox = function() {

        var dom_box = $(document.createElement('input'));

        dom_box.attr('type', "checkbox");
        dom_box.attr('class', "form-control");

        return dom_box;
    };


    self.create_button = function() {

        var dom_button = $(document.createElement('button'));

        dom_button.attr('class', "btn");

        return dom_button;
    };

    self.create_switch_button = function(callback, id_name) {

        var button = self.create_button(),
            fun = callback || (function(state) { console.log(state); }),
            id = id_name || "switch";

        button.attr('id', id);

        //Behavior of switch button
        button.click(function() {
            var state_table = {true: {'class': 'btn-danger',
                                      'label': 'ON'},
                               false: {'class': 'btn-success',
                                       'label': 'OFF'}},
                me = $(this),
                state = me.hasClass(state_table[true]['class']);

            me.removeClass(state_table[state]['class']);
            state = !state;
            me.addClass(state_table[state]['class']);
            me.text(state_table[state]['label']);
            fun(state);
        });

        button.addClass('btn-success');
        button.text('OFF');

        return button;
    };

    self.init_notification_box = function() {

        self.dom_notify.click(function(event) {
            $(this).animate({opacity: 0}, 1000);
        });
    };

    self.notification_event = function(kind, text, duration, fade_duration) {

        fade_duration = fade_duration || 1000;
        duration = duration || 0;

        self.dom_notify.attr('class', 'alert alert-' + kind);
        self.dom_notify.text(text);
        self.dom_notify.css({opacity: 1.0});

        if (duration > 0) {
            setTimeout(function() {
                self.dom_notify.css({opacity: 1.0}).animate({opacity: 0.0}, fade_duration)
            }, duration);
        }
    };

    self.download_url  = function(url, filename) {
		var link = window.document.createElement('a');
		link.href = url;
		link.download = filename || 'image.png';
		var click = document.createEvent("Event");
		click.initEvent("click", true, true);
		link.dispatchEvent(click);
	};

    self.assign_cookie = function(widget, cookie_name, default_value) {

        var value;

        if (Cookies.get(cookie_name) == null) {
            value = default_value;
        }
        else {
            value = Cookies.get(cookie_name)
        }
        widget.val(value);

        widget.change(function(event) {
            Cookies.set(cookie_name, widget.val());
        });
    };
};
