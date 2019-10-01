#!/bin/bash -u

main()
{
    sockets_spec=${1:-"ipc:///dev/shm/cambiar-este-atributo-por-el-nuevo-IPC"}; shift
    stage_dir=${1:-"./thinsec/images"}; shift
    driver_script_server=${1:-"./message_middleware/server_zmq"}; shift
    driver_script_client=${1:-"./message_middleware/client_zmq"}; shift
    local_port=${1:-5000}; shift

    cleanup
    trap cleanup EXIT

    python3 ${sockets_spec} ${local_port} &
    ${driver_script_dir}/bin/release/flyzmqserver ${sockets_spec} ${stage_dir} &
    ${driver_script_dir}/frame_observer.py ${sockets_spec}.vid &
    sleep 2 && firefox "http://localhost:${local_port}" &

    echo "My PID is $$ (Use kill -KILL $$ to terminate all processes)"
    wait

}

cleanup()
{
    echo "Removing stale sub-apps" >&2
    pkill -KILL flyzmqclient
    pkill -KILL -f $(basename ${gui_script})
}

main "$@"
