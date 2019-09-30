#!/bin/bash -u

main()
{
    sockets_spec=${1:-"ipc:///dev/shm/pgrcam"}; shift
    stage_dir=${1:-"$HOME/image/stage"}; shift
    gui_script=${1:-"$HOME/code/thinsec/microscope/main_server.py"}; shift
    driver_script_dir=${1:-"$HOME/code/pgr"}; shift
    local_port=${1:-5000}; shift

    cleanup
    trap cleanup EXIT

    python2 ${gui_script} -w 50 ${sockets_spec} ${local_port} &
    ${driver_script_dir}/server-zmq/bin/release/flyzmqserver ${sockets_spec} ${stage_dir} &
    ${driver_script_dir}/client-zmq/frame_observer.py ${sockets_spec}.vid &
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
