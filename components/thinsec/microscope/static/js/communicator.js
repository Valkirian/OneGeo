var Comm = new function() {

    var self = this;

    self.on_open = undefined;
    self.on_close = undefined;
    self.on_message = undefined;
    self.on_error = undefined;

    self.start = function(websocket_URI) {
        self.ws = new WebSocket(websocket_URI);
        self.ws.binaryType = "arraybuffer";
        self.ws.onopen = self.on_open;
        self.ws.onclose = self.on_close;
        self.ws.onmessage = self.on_message;
        self.ws.onerror = self.on_error;
    };

    self._send = function(message, name, type)
    {
        var msg = { 'message': message,
                    'name': name,
                    'type': type };
        self.ws.send(JSON.stringify(msg));
    };

    self.send_command = function(name, command, params)
    {
        var payload = {'command': command};
        if (params) payload['params'] = params;

        self._send(payload, name, 'control');
    };

};
