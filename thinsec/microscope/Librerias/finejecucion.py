# Finalizando la ejecucion
def shutdown():
    for motor in motor_api.motor_address.keys():
        motor_execution.set_hold_force(motor, False)
        gevent.sleep(0.01)
        #motor_api.py
    motor_execution.set_axes_enabled(False)
    server.stop()
    zmq_ctx.destroy()
    sys.exit()