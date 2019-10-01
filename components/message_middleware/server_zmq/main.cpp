#include "main.h"

static const char USAGE[] =
R"(FlyCapture ZeroMQ camera server.

    Usage:
      flyzmqserver <socket-spec> <save-dir> [--report-period=<secs>] [--compression=<level>]
      flyzmqserver (-h | --help)

    Options:
      -h --help                           Show this screen.
      -r <secs>, --report-period=<secs>   Interval of periodic status messages  [default: 0.5].
      -c <level>, --compression=<level>   Compression level for saving images   [default: 2].
)";

DebugCerr debug_cerr;   // cerr wrapped with a timestamp

zmqpp::context zmq_ctx;
zmqpp::socket cmd_pull(zmq_ctx, zmqpp::socket_type::pull);          //Handles incoming camera config comands
zmqpp::socket state_pub(zmq_ctx, zmqpp::socket_type::pub);          //Emits status updates
zmqpp::socket video_pub(zmq_ctx, zmqpp::socket_type::pub);          //Emits captured camera frames
zmqpp::socket image_push(zmq_ctx, zmqpp::socket_type::push);        //Transports image save data to writter threads
zmqpp::socket worker_cmd_push(zmq_ctx, zmqpp::socket_type::push);   //Emits command to worker threads
zmqpp::socket worker_ret_pull(zmq_ctx, zmqpp::socket_type::pull);   //Receives result messages from worker threads
zmqpp::socket imager_cmd_push(zmq_ctx, zmqpp::socket_type::push);   //Emits command tothe imager thread 
const string image_push_addr("inproc://wkiph");
const string worker_cmd_addr("inproc://wkcpb");
const string worker_ret_addr("inproc://wkrph");
const string imager_cmd_addr("inproc://imrph");

PNGOption save_settings;
fs::path sequence_dir;
atomic<unsigned int> axis_idx;
atomic<bool> running, in_capture, fastaxis_pending;
vector<string> fastaxis_sequence;
vector<double> fastaxis_sequence_expos;

vector<thread> workers;


int main(int argc, char **argv)
{
    auto camdrv = make_shared<CameraDriver>(unique_ptr<Camera>(new Camera()));
    Error error;

    duration report_period;
    string zmq_socket_spec;
    fs::path base_directory;
    int compression_level;
	optmap args = docopt::docopt(USAGE, { argv + 1, argv + argc });
    parse_command_line(args, &zmq_socket_spec, &report_period, &base_directory, compression_level);

    // PNG saving options for images in the sequences
    save_settings.compressionLevel = compression_level;
    save_settings.interlaced = false;

    // Check if save directory is valid
    if (!fs::exists(base_directory)) {
        debug_cerr << "Specified save directory does not exist or has insufficient permissions" << endl;
        return false;
    }

    if (!bind_sockets(zmq_socket_spec)) {
        debug_cerr << "Unable to bind ZMQ sockets" << endl;
        return false;
    }
    //Wait time for pub socket to initialize
    zmqpp::poller poller_sockets;
	poller_sockets.add(cmd_pull, zmqpp::poller::poll_in);
    poller_sockets.add(worker_ret_pull, zmqpp::poller::poll_in);

    thread imager(image_grab_thread, camdrv);
    int available_threads = thread::hardware_concurrency() - 2;
    for (int i=0; i < max(available_threads, 1); ++i) {
        workers.push_back(thread(image_writer_thread));
    }
    cerr << "Using " << workers.size() << " saver threads" << endl;

    signal(SIGINT, handler_terminate_program);
    signal(SIGTERM, handler_terminate_program);

    time_point now, last = clock_src::now();
    last = clock_src::now();
    in_capture = fastaxis_pending = false;
    running = true;
    while(running)
    {
        now = clock_src::now();
        if ((now - last) >= report_period) {
            last += report_period;
            socket_send(state_pub, camdrv->get_config_json(), "_cfg_");
        }

        if (poller_sockets.poll(5)) {
            zmqpp::message message;

            if (poller_sockets.has_input(cmd_pull)) {
                cmd_pull.receive(message);
                string raw, parse_err;
                message >> raw;
                Json cmd = Json::parse(raw, parse_err);
                handle_command(cmd, base_directory, camdrv);
            }

            if (poller_sockets.has_input(worker_ret_pull)) {
                worker_ret_pull.receive(message);
                string to_send;
                message >> to_send;
                state_send(to_send);
            }
        }
    }

    debug_cerr << "Closing..." << endl;

    for (auto i=workers.size(); i>0; --i) worker_cmd_push.send("exit");
    for (auto &t : workers) t.join();
    imager_cmd_push.send("exit");
    imager.join();

    camdrv->disconnect();

    cmd_pull.close();
    state_pub.close();
    video_pub.close();
    worker_ret_pull.close();
    image_push.close();
    worker_cmd_push.close();
    imager_cmd_push.close();

    return 0;
}

void parse_command_line(optmap &args, string *socket_spec, duration *report_period, fs::path *base_dir, int &comp_level)
{
    /* Debugging docopt parsed names
    for (auto& kv: args) {
        cout << kv.first << ": " << kv.second << endl;
    }
    */

    // docopt::value does not convert to basic types
    double period;
    istringstream s_p(args["--report-period"].asString());
    s_p >> period;
    duration gen_period((int)(period * 1e3));

    *report_period = gen_period;

    istringstream s_c(args["--compression"].asString());
    s_c >> comp_level;

    *socket_spec = args["<socket-spec>"].asString();

    *base_dir = args["<save-dir>"].asString();

    cout << "Specified options: " << endl
         << "-ZMQ socket: " << *socket_spec << endl
         << "-report period: " << period << "s" << endl
         << "-compression level: " << comp_level << endl
         << "-image sequence save dir: " << *base_dir << endl;
}

bool bind_sockets(const string &socket_basename)
{
    stringstream cmd_socket_addr, state_socket_addr, video_socket_addr;

    if (socket_basename.find("ipc://", 0) != string::npos) {
        cmd_socket_addr << socket_basename << ".cmd";
        state_socket_addr << socket_basename << ".stt";
        video_socket_addr << socket_basename << ".vid";
    }
    else if (socket_basename.find("tcp://", 0) != string::npos) {
        unsigned int index = socket_basename.rfind(':');
        if (index == 3) {
            debug_cerr << "TCP socket address malformed" << endl;
            return false;
        }
        istringstream conv_ss(socket_basename.substr(index + 1));
        string base_addr = socket_basename.substr(0, index);
        unsigned int base_port;
        conv_ss >> base_port;
        cmd_socket_addr << base_addr << ':' << base_port;
        state_socket_addr << base_addr << ':' << base_port + 1;
        video_socket_addr << base_addr << ':' << base_port + 2;
    }
    else {
        cout << "It is not a supported transport type" << endl;
        return false;
    }

    cmd_pull.bind(cmd_socket_addr.str());
    state_pub.bind(state_socket_addr.str());
    video_pub.bind(video_socket_addr.str());
    image_push.bind(image_push_addr);
    worker_cmd_push.bind(worker_cmd_addr);
    worker_ret_pull.bind(worker_ret_addr);
    imager_cmd_push.bind(imager_cmd_addr);

    cout << "   command socket: " << cmd_socket_addr.str() << endl
         << "   state socket: " << state_socket_addr.str() << endl
         << "   video socket: " << video_socket_addr.str() << endl;

    return true;
}

bool submit_image_for_save(Image *image, string filepath)
{
    zmqpp::message payload;

    int len = image->GetDataSize();
    string image_buffer(len, ' ');

    copy_n(image->GetData(), len, image_buffer.begin());

    payload << filepath << image_buffer
            << image->GetRows() << image->GetCols() << image->GetStride()
            << (unsigned int)(image->GetPixelFormat())
            << (unsigned int)(image->GetBayerTileFormat());

    return image_push.send(payload);
}

void send_cv2_frame(Image *pImage)
{
    // convert to rgb
    unique_ptr<Image> temp_image(new Image);
    pImage->Convert( PIXEL_FORMAT_BGR, temp_image.get() );

    // convert to OpenCV Mat
    unsigned int rowBytes = (double)temp_image->GetReceivedDataSize()/(double)temp_image->GetRows();
    unique_ptr<cv::Mat> image(new cv::Mat(temp_image->GetRows(), temp_image->GetCols(),
                                          CV_8UC3, temp_image->GetData(), rowBytes));

    // https://github.com/quanhua92/LearnZeroMQ/blob/master/14_Send_OpenCV_Mat/
    socket_send(video_pub, image->total() * image->channels(),
                (const char*)image->data);
}

void image_grab_thread(shared_ptr<CameraDriver> camdrv)
{
    zmqpp::socket command_pull(zmq_ctx, zmqpp::socket_type::pull);
    zmqpp::socket result_push(zmq_ctx, zmqpp::socket_type::push);
    command_pull.connect(imager_cmd_addr);
    result_push.connect(worker_ret_addr);
    zmqpp::poller poller;
    poller.add(command_pull, zmqpp::poller::poll_in);
    unique_ptr<Image> grabbed(new Image);

    if (!camdrv->initialize(0, 0, MODE_0, PIXEL_FORMAT_RAW12, HQ_LINEAR)) {
        debug_cerr << "Unable to initialize camera" << endl;
        return;
    }
    camdrv->set_trigger(false);

    if (!camdrv->start()) {
        return;
    }
    camdrv->set_strobe(1, false);

    while (true)
    {
        if (camdrv->get_image(grabbed.get())) {
            if (in_capture && fastaxis_pending) {
                submit_image_for_save(grabbed.get(), fastaxis_sequence[axis_idx]);
                if (++axis_idx == fastaxis_sequence.size()) {
                    fastaxis_pending = false;
                }
                else {
                    debug_cerr << "Setting shutter to " << fastaxis_sequence_expos[axis_idx] << endl;
                    camdrv->set_by_key("shutter_speed", fastaxis_sequence_expos[axis_idx]);
                }
            }
            else {
                send_cv2_frame(grabbed.get());
            }
        }

        if (poller.poll(0)) {
            zmqpp::message msg_in;
            command_pull.receive(msg_in);
            string command;
            msg_in >> command;
            if (command == "exit") {
                break;
            }
            else {
                if (in_capture) {
                    string parse_err;
                    Json cmd = Json::parse(command, parse_err);
                    fastaxis_pending = true;
                    axis_idx = 0;

                    fastaxis_sequence.clear();
                    for (auto const &item: cmd["seq"].array_items()) {
                        fastaxis_sequence.push_back(item.string_value());
                    }

                    fastaxis_sequence_expos.clear();
                    for (auto const &item: cmd["expo"].array_items()) {
                        fastaxis_sequence_expos.push_back(item.number_value());
                    }

                    debug_cerr << "Preparing shutter to " << fastaxis_sequence_expos[0] << endl;
                    camdrv->set_by_key("shutter_speed", fastaxis_sequence_expos[0]);

                    int leftover = prepare_for_sweep(camdrv);
                    debug_cerr << "Leftover images in memory: " << leftover << endl;

                    json_send(result_push, Json::object{{"ready", true}});
                }
            }
        }
    }

    camdrv->stop();
    camdrv->set_power(false);
}

void image_writer_thread()
{
    zmqpp::socket image_pull(zmq_ctx, zmqpp::socket_type::pull);
    zmqpp::socket command_pull(zmq_ctx, zmqpp::socket_type::pull);
    zmqpp::socket result_push(zmq_ctx, zmqpp::socket_type::push);
    image_pull.connect(image_push_addr);
    command_pull.connect(worker_cmd_addr);
    result_push.connect(worker_ret_addr);

    zmqpp::poller poller;
    poller.add(image_pull, zmqpp::poller::poll_in);
    poller.add(command_pull, zmqpp::poller::poll_in);

    while (true)
    {
        if (poller.poll(5)) {
            zmqpp::message msg_in;

            if (poller.has_input(image_pull)) {
                image_pull.receive(msg_in);

                string save_filepath;
                unsigned int img_rows, img_cols, img_stride, img_f_pix, img_f_bay;
                string image_buffer;
                msg_in >> save_filepath >> image_buffer
                       >> img_rows >> img_cols >> img_stride >> img_f_pix >> img_f_bay;
                Image saved_image(img_rows, img_cols, img_stride,
                                  (unsigned char*)(image_buffer.data()),
                                  image_buffer.size(), (PixelFormat)img_f_pix,
                                  (BayerTileFormat)img_f_bay);
                save_image(saved_image, save_filepath, result_push);
            }

            if (poller.has_input(command_pull)) {
                command_pull.receive(msg_in);
                string command;
                msg_in >> command;
                if (command == "exit") {
                    return;
                }
            }
        }
    }
}

void save_image(Image &temp_image, string filename, zmqpp::socket &socket)
{
    stringstream filenom;
    filenom << filename << ".png";
    fs::path filepath = sequence_dir / filenom.str();
    debug_cerr << "Saving image to " << filepath << endl;
    json_send(socket, Json::object{{"saving", filepath.c_str()}});

    // convert to rgb
    unique_ptr<Image> rgbImage(new Image);
    //temp_image.SetColorProcessing(DIRECTIONAL_FILTER);
    temp_image.Convert( PIXEL_FORMAT_BGR, rgbImage.get() );

    Error error = rgbImage->Save(filepath.c_str(), &save_settings);
    if (error != PGRERROR_OK) {
        json_send(socket, Json::object{{"save-error", filepath}});
        debug_cerr << "Error saving image " << filepath << ":" << endl;
        PrintError(error);
    }
}

bool handle_command(Json &cmd, fs::path &base_directory, shared_ptr<CameraDriver> camdrv)
{
    string kind = cmd[0].string_value();
    Json command = cmd[1];

    debug_cerr << "Command " << kind << ": " << command.dump() << endl;

    if (kind == "config") {
        debug_cerr << command.dump() << endl;
        for (auto const &item: command.object_items()) {
            string key = item.first;
            if (CameraDriver::is_property_float(key)) {
                camdrv->set_by_key(key, command[key].number_value());
            }
            else {
                if (command[key].is_array()){
                    camdrv->set_by_key(key, command[key][0].int_value(),
                                            command[key][1].int_value());
                }
                else {
                    camdrv->set_by_key(key, command[key].int_value(), 0);
                }
            }
        }
    }
    else if (kind == "state") {
        for (auto const &item: command.object_items()) {
            string key = item.first;
            if (key == "power") {
                bool power = command[key].bool_value();
                if (power) {
                    camdrv->set_power(power);
                    camdrv->set_strobe(1, false);
                    camdrv->start();
                }
                else {
                    camdrv->stop();
                    camdrv->set_power(power);
                }
            }
        }
    }
    else if (kind == "sweep") {
        in_capture = command["active"].bool_value();

        if (in_capture) {
            sequence_dir = base_directory / command["name"].string_value();

            if (fs::is_directory(sequence_dir)) {
                debug_cerr << "Directory " << sequence_dir << " already exists. "
                           << "Sweep cancelled." << endl;
                in_capture = false;
            }
            else {
                if (!fs::create_directory(sequence_dir)) {
                    debug_cerr << "Unable to create directory " << sequence_dir
                               << ". Sweep cancelled." << endl;
                    in_capture = false;
                }
                else {
                    fs::permissions(sequence_dir, fs::perms::add_perms | fs::perms::group_write);
                    debug_cerr << "Created sweep directory " << sequence_dir << endl;
                    fs::path target_dir;
                    for (auto const &dir_name: command["directories"].array_items()) {
                        target_dir = sequence_dir / dir_name.string_value();
                        fs::create_directories(target_dir);
                        debug_cerr << "Created sub-directory " << target_dir << endl;
                    }
                }
            }
        }
        // Error checking might have set in_capture to false
        if (in_capture) {
            camdrv->set_trigger(0, 0, 1);
            camdrv->set_strobe(1, 1, 0, 0);
        }
        else {
            camdrv->set_trigger(false);
            camdrv->set_strobe(1, false);
        }
        fastaxis_pending = false;
        state_send(Json::object{ {"sweep", (bool)in_capture} });

        debug_cerr << "Sweep ";
        if (!in_capture) cerr << "in";
        cerr << "active" << endl;
    }
    else if (kind == "fastaxis") {
        json_send(imager_cmd_push, command);
    }
    else {
        debug_cerr << "Unknown command of kind (" << kind << "): " << command.dump() << endl;
        return false;
    }

    return true;
}

bool socket_send(zmqpp::socket &socket, unsigned int length, const char *buffer)
{
    zmqpp::message message;
    message.add_raw(buffer, length);
    return socket.send(message);
}

bool socket_send(zmqpp::socket &socket, const Json &object, const string &header)
{
    zmqpp::message message;
    message << header + object.dump();
    return socket.send(message);
}

bool socket_send(zmqpp::socket &socket, const string &msg, const string &header)
{
    zmqpp::message message;
    message << header + msg;
    return socket.send(message);
}

bool state_send(const Json &object)
{
    return socket_send(state_pub, object, "_stt_");
}

bool state_send(const string &tosend)
{
    return socket_send(state_pub, tosend, "_stt_");
}

bool json_send(zmqpp::socket &sock, const Json &object)
{
    zmqpp::message message;
    message << object.dump();
    return sock.send(message);
}

int prepare_for_sweep(shared_ptr<CameraDriver> camdrv)
{
    unique_ptr<Image> _toss(new Image);
    int counter = 0;

    for (int i=0; i < 5; ++i) {
        this_thread::sleep_for(chrono::milliseconds(100));
        while (camdrv->get_image(_toss.get())) {
            ++counter;
        }
    }

    return counter;
}

void handler_terminate_program(int sig)
{
    running = false;
}
