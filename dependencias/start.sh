#!/bin/bash -u

main()
{   
    # Establenciendo las variables de entorno y los shared memory
    gui_script=${1:-"main_server.py"}
    sockets_spec=${1:-"ipc:///dev/shm/"}; shift
    stage_dir=${1:-"./images"}; shift
    driver_script_server=${1:-"./message_middleware/server_zmq"}; shift
    driver_script_client=${1:-"./message_middleware/client_zmq"}; shift
    local_port=${1:-5000}; shift

    cleanup
    trap cleanup EXIT

    # Haciendo la ejecucion del main server, abriendo el mavegador y abriendo los sockets.
    python3 ${gui_script} -w 50 ${sockets_spec} ${local_port} &
    ${driver_script_server}/bin/release/flyzmqserver ${sockets_spec} ${stage_dir} &
    ${driver_script_client}/frame_observer.py ${sockets_spec}.vid &
    sleep 2 && google-chrome-stable --no-sandbox "http://localhost:${local_port}" &

    echo "My shmid is $$ (Use kill -KILL $$ to terminate all processes)"
    wait

}

cleanup()
{
    echo "Removing stale sub-apps" >&2
    pkill -KILL flyzmqclient
    pkill -KILL -f $(basename ${gui_script})
}

main "$@"
