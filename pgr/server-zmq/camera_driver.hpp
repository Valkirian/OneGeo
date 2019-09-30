#ifndef __CAMDRIVER_H__
#define __CAMDRIVER_H__

#include <flycapture/FlyCapture2.h>

#include <chrono>
#include <ctime>
#include <math.h>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "json11.hpp"

#define _COLS           2448
#define _ROWS           2048
#define _BYTES_PER_PIXEL  2
#define _IMAGE_SIZE     _COLS * _ROWS * _BYTES_PER_PIXEL


using namespace FlyCapture2;
using namespace std;
using namespace json11;

class CameraDriver
{
    unique_ptr<Camera>               camera;
    static map<string, PropertyType> floats_property_map;
    static map<string, PropertyType> ints_property_map;
    vector<unsigned char>            api_buffer;
    bool                             capturing;

    public:

        CameraDriver(unique_ptr<Camera>);

        bool is_powered();
        bool set_power(bool);
        bool initialize(unsigned int, unsigned int, Mode, PixelFormat, ColorProcessingAlgorithm);
        bool start(ImageEventCallback callbackFn = NULL);
        bool stop();
        bool disconnect();

        bool get_image(Image*);
        string get_config_json();

        static bool is_property_float(string&);
        bool set_by_key(const string&, double);
        bool set_by_key(const string&, int, int);

        bool set_trigger(bool);
        bool set_trigger(unsigned int, unsigned int, unsigned int);
        bool set_strobe(unsigned int, bool);
        bool set_strobe(unsigned int, unsigned int, double, double);

        bool poll_for_trigger_ready();
};

void PrintError(Error);

#endif //__CAMDRIVER_H__
