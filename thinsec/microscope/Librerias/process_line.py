def process_command_line():

    description = "Main program for controlling an automated microscope"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "camera_server_address",
        default="tcp://localhost:50000",
        help="socket base address where camera server is listening",
    )
    parser.add_argument(
        "port", type=int, default=5000, help="TCP port over which to listen"
    )
    parser.add_argument(
        "-u",
        "--motor-debug",
        action="store_true",
        help=("If present, data traffic to/from motor controller will " "be shown"),
    )

    return parser.parse_args()
