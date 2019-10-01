#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>

#include <boost/filesystem.hpp>
#include <flycapture/FlyCapture2.h>
#include <opencv2/core/core.hpp>
#include <zmqpp/zmqpp.hpp>

#include "ctpl_stl.h"
#include "docopt.h"
#include "json11.hpp"

#include "camera_driver.hpp"
#include "utils.hpp"

using namespace FlyCapture2;
using namespace std;
using namespace json11;
namespace fs = boost::filesystem;

using optmap = map<string, docopt::value>;
using clock_src = chrono::high_resolution_clock;
using time_point = chrono::time_point<clock_src>;
using duration = chrono::duration<long int, milli>; 

void parse_command_line(optmap&, string*, duration*, fs::path*, int&);
bool bind_sockets(const string&);
bool handle_command(Json&, fs::path&, shared_ptr<CameraDriver>);
int prepare_for_sweep(shared_ptr<CameraDriver>);
void send_cv2_frame(Image*);
void save_image(Image&, string, zmqpp::socket&);
bool submit_image_for_save(Image*, int, int, int);
void image_writer_thread();
void image_grab_thread(shared_ptr<CameraDriver>);
bool socket_send(zmqpp::socket&, unsigned int, const char*);
bool socket_send(zmqpp::socket&, const Json&, const string&);
bool socket_send(zmqpp::socket&, const string&, const string&);
bool json_send(zmqpp::socket&, const Json&);
bool state_send(const Json&);
bool state_send(const string&);
void handler_terminate_program(int);
