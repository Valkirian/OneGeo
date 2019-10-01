var UXKeyboard = new function() {

    var self = this;

    self.press_callbacks = {};

    self.field_is_numeric = function(field) {
        field.keydown(self.input_only_numbers);
        //field.attr('type', 'number');
        self.desensitize_in_field(field);
    };

    self.input_only_numbers = function(e) {
        // Allow: backspace, delete, tab, escape, enter, -, +, ',' and .
        if ($.inArray(e.keyCode, [8, 9, 13, 27, 32, 46, 107, 109, 110, 187, 188, 189, 190]) !== -1 ||
                // Allow: Ctrl+A
            (e.keyCode == 65 && e.ctrlKey === true) ||
                // Allow: Ctrl+C
            (e.keyCode == 67 && e.ctrlKey === true) ||
                // Allow: Ctrl+X
            (e.keyCode == 88 && e.ctrlKey === true) ||
                // Allow: home, end, left, right
            (e.keyCode >= 35 && e.keyCode <= 39)) {
                    // let it happen, don't do anything
                    return;
        }
        // Ensure that it is a number and stop the keypress
        if ((e.shiftKey || (e.keyCode < 48 || e.keyCode > 57)) && (e.keyCode < 96 || e.keyCode > 105)) {
            e.preventDefault();
        }
    };

    self.desensitize_in_field = function(field) {
        field.focusin(function() {
            self.actions_enabled(false);
        });
        field.focusout(function() {
            self.actions_enabled(true);
        });
    };

    self.actions_enabled = function(enabled) {
        if (enabled) {
            $(document).keyup(self.keyboard_keyup);
            $(document).keydown(self.keyboard_keydown);
            $(document).keypress(self.keyboard_keypress);
        }
        else {
            $(document).off('keypress');
            $(document).off('keyup');
            $(document).off('keydown');
        }
    };

    self.keyboard_keydown = function(event) {
        console.info(['dn', {key: event.key, which: event.which, code: event.keyCode}]);
    };

    self.keyboard_keyup = function(event) {
        console.info(['up', {key: event.key, which: event.which, code: event.keyCode}]);
    };

    self.keyboard_keypress = function(event) {
        console.info(['pr', {key: event.key, which: event.which, code: event.keyCode}]);

        if (self.press_callbacks.hasOwnProperty(event.which)) {
            self.press_callbacks[event.which]();
        }
    };

    self.register_keypress = function(keycodes, callback) {
        var target;

        if ($.isNumeric(keycodes)) {
            target = [keycodes];
        }
        else {
            target = keycodes;
        }

        $.each(target, function(index, value) {
            if (value in self.press_callbacks) {
                throw "Key press callback already registered for keycode " + value
            }
            else {
                self.press_callbacks[value] = callback;
            }
        });
    };
};
