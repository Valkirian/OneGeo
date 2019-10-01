#include "camera_driver.hpp"

CameraDriver::CameraDriver(unique_ptr<Camera> cam)
{
    camera = move(cam);
    capturing = false;
}

bool CameraDriver::initialize(unsigned int width, unsigned int height, Mode mode, PixelFormat pixel_format, ColorProcessingAlgorithm image_algorithm)
{
    Error error;

    BusManager busMgr;
    unsigned int numCameras;
    error = busMgr.GetNumOfCameras(&numCameras);
    if (error != PGRERROR_OK) {
        PrintError( error );
        return false;
    }

    cout << "Number of cameras detected: " << numCameras <<endl;

    if ( numCameras < 1 )
    {
        cerr << "Insufficient number of cameras... exiting" << endl;
        return false;
    }

    PGRGuid guid;
    error = busMgr.GetCameraFromIndex(0, &guid);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return false;
    }

    // Connect to a camera
    error = camera->Connect(&guid);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return false;
    }
    else {
        cerr << "Connected to camera" << endl;
    }

    // Power on the camera
    if (!set_power(true)) {
        cerr << "Failed to power on the camera" << endl;
        PrintError(error);
        return false;
    }

    // Get the camera info and print it out
    CameraInfo camInfo;
    error = camera->GetCameraInfo( &camInfo );
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to get info from camera" << endl;
        PrintError(error);
        return false;
    }
    FC2Config camConfig;
    error = camera->GetConfiguration( &camConfig );
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to get config from camera" << endl;
        PrintError(error);
        return false;
    }
    cout << camInfo.vendorName << " "
         << camInfo.modelName << " "
         << camInfo.serialNumber << endl;

    Format7Info fmt7Info;
    fmt7Info.mode = mode;
	bool isSupported;
    error = camera->GetFormat7Info( &fmt7Info, &isSupported );

    if ( (pixel_format & fmt7Info.pixelFormatBitField) == 0 )
    {
        // Pixel format not supported!
		cerr << "Pixel format is not supported" << endl;
        return false;
    }
    if ( isSupported == false )
    {
        // format 7 mode is not valid for this camera
		cerr << "The format 7 mode is not valid for this camera" << endl;
        return false;
    }

    width = (width == 0) ? fmt7Info.maxWidth : width;
    height = (height ==0) ? fmt7Info.maxHeight : height;
    if ( width > fmt7Info.maxWidth || height > fmt7Info.maxHeight)
    {
        // ROI is out of bounds!
		cerr << "ROI is out of bounds!" << endl;
        return false;
    }

    Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = mode;
    fmt7ImageSettings.offsetX = 0;
    fmt7ImageSettings.offsetY = 0;
    fmt7ImageSettings.width = width;
    fmt7ImageSettings.height = height;
    fmt7ImageSettings.pixelFormat = pixel_format;

    bool isValid;
    Format7PacketInfo fmt7PacketInfo;

    // Validate the settings to make sure that they are valid
    error = camera->ValidateFormat7Settings(&fmt7ImageSettings, &isValid,
                                            &fmt7PacketInfo);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return false;
    }

    if ( !isValid )
    {
        // Settings are not valid
		cerr << "Format7 settings are not valid" << endl;
        return false;
    }

    // Set the settings to the camera
    unsigned int bytesPerPacket = fmt7PacketInfo.recommendedBytesPerPacket;
    error = camera->SetFormat7Configuration(&fmt7ImageSettings,
                                            bytesPerPacket);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return false;
    }

    Image::SetDefaultColorProcessing(image_algorithm);

    camConfig.grabMode = DROP_FRAMES;
    camConfig.grabTimeout = 2;
    camConfig.highPerformanceRetrieveBuffer = false;
    camConfig.registerTimeoutRetries = 1;
    camConfig.numBuffers = 10;
    camConfig.numImageNotifications = 1;

    error = camera->SetConfiguration( &camConfig );
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to set config to camera" << endl;
        PrintError(error);
        return false;
    }

    // 1. Calculate the user buffers sizes on packet boundary
    unsigned int numOfPackets = static_cast<unsigned int>(ceil(( _IMAGE_SIZE) / static_cast<double>(bytesPerPacket)));
    unsigned int imageSize = numOfPackets * bytesPerPacket;
    // 2. Allocate the memory for the user buffers
    api_buffer.reserve(imageSize * camConfig.numBuffers);
    // 3. Set the new allocated memory as the user buffers that the camera will fill
    error = camera->SetUserBuffers(api_buffer.data(), imageSize, camConfig.numBuffers);
    if( error != PGRERROR_OK )
    {
        cerr << "Failed to get config from camera" << endl;
        PrintError(error);
        return false;
    }

    error = camera->GetConfiguration( &camConfig );
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to get config from camera" << endl;
        PrintError(error);
        return false;
    }
    map<GrabMode, string> grab_modes = {
        {DROP_FRAMES, "drop_frames"},
        {BUFFER_FRAMES, "buffer_frames"},
        {UNSPECIFIED_GRAB_MODE, "unspecified"}
    };
    cerr << "grab mode: " << grab_modes[camConfig.grabMode] << endl
         << "grab timeout: " << camConfig.grabTimeout << endl
         << "high perf: " << camConfig.highPerformanceRetrieveBuffer << endl
         << "register retries: " << camConfig.registerTimeoutRetries << endl
         << "num buffs: " << camConfig.numBuffers << endl
         << "num Notifies: " << camConfig.numImageNotifications << endl
         << "min num notifies: " << camConfig.minNumImageNotifications << endl;
    return true;
}

bool CameraDriver::is_powered()
{
    const unsigned int r_cameraPower = 0x610;
    unsigned int powerVal;
    Error error = camera->ReadRegister(r_cameraPower, &powerVal);
    if (error != PGRERROR_OK) {
        PrintError( error );
        return false;
    }
    return ((powerVal & 0x80000000) != 0);
}

bool CameraDriver::set_power(bool on)
{
    const unsigned int r_cameraPower = 0x610;
    const unsigned int k_powerVal = 0x80000000;
    Error error = camera->WriteRegister(r_cameraPower, on? k_powerVal : 0);
    if (error != PGRERROR_OK) {
        PrintError( error );
        return false;
    }

    if (on) {    // Wait for camera to complete power-up
        const unsigned int millisecondsToSleep = 100;
        unsigned int regVal = 0;
        unsigned int retries = 10;
        do
        {
            this_thread::sleep_for(chrono::milliseconds(millisecondsToSleep));
            error = camera->ReadRegister(r_cameraPower, &regVal);
            if (error == PGRERROR_TIMEOUT)
            {
                // ignore timeout errors, camera may not be responding to
                // register reads during power-up
            }
            else if (error != PGRERROR_OK) {
                PrintError( error );
                return false;
            }
            else

            retries--;
        } while ((regVal & k_powerVal) == 0 && retries > 0);

        // Check for timeout errors after retrying
        if (error == PGRERROR_TIMEOUT) {
            PrintError( error );
            return false;
        }
    }

    return true;
}

bool CameraDriver::start(ImageEventCallback callbackFn)
{
    bool retcode = false;

    Error error = camera->StartCapture(callbackFn);
    if ( error == PGRERROR_OK ) {
        retcode = true;
    }
    else if ( error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED ) {
        cerr << "Bandwidth exceeded" << endl;
    }
    else {
        cerr << "Failed to start image capture" << endl;
    }

    capturing = retcode;
    return retcode;
}

bool CameraDriver::stop()
{
    bool retcode = false;

    if (capturing) {
        if ( camera->StopCapture() == PGRERROR_OK ) {
            retcode = true;
        }
        else {
            cerr << "Could not stop image capture" << endl;
        }
    }

    capturing = !retcode;
    return retcode;
}

bool CameraDriver::disconnect()
{
    if ( camera->Disconnect() != PGRERROR_OK ) {
        cerr << "Could not disconnect" << endl;
        return false;
    }
    return true;
}

string CameraDriver::get_config_json()
{
    double brightness, exposure_ev, shutter_ms, gain_db, sharpness,
           saturation, hue, temperature_celcius;
    int whitebal_blue, whitebal_red;
    Property prop;
    Error error;

    prop.type = BRIGHTNESS;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read brightness" << endl;
        return "{}";
    }
    brightness = prop.absValue;

    prop.type = AUTO_EXPOSURE;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read auto-exposure EV" << endl;
        return "{}";
    }
    exposure_ev = prop.absValue;

    prop.type = SHUTTER;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read shutter time" << endl;
        return "{}";
    }
    shutter_ms = prop.absValue;

    prop.type = GAIN;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read gain" << endl;
        return "{}";
    }
    gain_db = prop.absValue;

    prop.type = SHARPNESS;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read sharpness" << endl;
        return "{}";
    }
    sharpness = prop.valueA;

    prop.type = SATURATION;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read saturation" << endl;
        return "{}";
    }
    saturation = prop.absValue;

    prop.type = HUE;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read Hue" << endl;
        return "{}";
    }
    hue = prop.absValue;

    prop.type = WHITE_BALANCE;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read " << endl;
        return "{}";
    }
    whitebal_red = prop.valueA;
    whitebal_blue = prop.valueB;

    prop.type = TEMPERATURE;
    error = camera->GetProperty(&prop);
    if ( error != PGRERROR_OK ) {
        cerr << "Failed to read temperature" << endl;
        return "{}";
    }
    temperature_celcius = (prop.valueA / 10.0) - 273.15;

    Json config = Json::object({
            {"brightness", brightness},
            {"sharpness", sharpness},
            {"saturation", saturation},
            {"auto_exposure", exposure_ev},
            {"shutter_speed", shutter_ms},
            {"gain_analog", gain_db},
            {"hue", hue},
            {"awb_gains", Json::array{whitebal_red, whitebal_blue}},
            {"temperature", temperature_celcius},
            {"power", is_powered()}
            });

    return config.dump();
}

bool CameraDriver::poll_for_trigger_ready()
{
    const unsigned int k_softwareTrigger = 0x62C;
    unsigned int regVal = 0;
    Error error;

    do
    {
        error = camera->ReadRegister(k_softwareTrigger, &regVal);
        if (error != PGRERROR_OK)
        {
            PrintError( error );
            return false;
        }

    } while ( (regVal >> 31) != 0 );

    return true;
}

bool CameraDriver::set_trigger(bool active)
{
    TriggerMode triggerMode;
    Error error;

    error = camera->GetTriggerMode( &triggerMode );
    if (error != PGRERROR_OK) {
        PrintError( error );
        cerr << "Unable to read trigger settings" << endl;
        return false;
    }
    triggerMode.onOff = active;
    error = camera->SetTriggerMode( &triggerMode );
    if (error != PGRERROR_OK) {
        PrintError( error );
        cerr << "Unable to write trigger settings" << endl;
        return false;
    }

    return true;
}

bool CameraDriver::set_trigger(unsigned int source, unsigned int mode, unsigned int polarity)
{
    TriggerMode triggerMode;
    Error error;

    triggerMode.onOff = true;
    triggerMode.parameter = 0;
    triggerMode.mode = mode;
    triggerMode.source = source;
    triggerMode.polarity = polarity;

    error = camera->SetTriggerMode( &triggerMode );
    if (error != PGRERROR_OK) {
        PrintError( error );
        cerr << "Unable to write trigger settings" << endl;
        return false;
    }

    return true;
}

bool CameraDriver::set_strobe(unsigned int pin, bool active)
{
    StrobeControl strobe_ctl;
    Error error;

    strobe_ctl.source = pin;
    error = camera->GetStrobe( &strobe_ctl );
    if (error != PGRERROR_OK) {
        PrintError( error );
        cerr << "Unable to read strobe settings" << endl;
        return false;
    }
    strobe_ctl.onOff = active;
    error = camera->SetStrobe( &strobe_ctl );
    if (error != PGRERROR_OK) {
        PrintError( error );
        cerr << "Unable to write strobe settings" << endl;
        return false;
    }

    return true;
}

bool CameraDriver::set_strobe(unsigned int source, unsigned int polarity, double delay, double duration)
{
    StrobeControl strobe_ctl;
    Error error;

    strobe_ctl.source = source;
    strobe_ctl.onOff = true;
    strobe_ctl.polarity = polarity;
    strobe_ctl.delay = delay;
    strobe_ctl.duration = duration;

    error = camera->SetStrobe( &strobe_ctl );
    if (error != PGRERROR_OK) {
        PrintError( error );
        cerr << "Unable to write strobe settings" << endl;
        return false;
    }

    return true;
}

// Map of properties whose effective value is that of member "absValue", a double
map<string, PropertyType> CameraDriver::floats_property_map = {
            {"brightness", BRIGHTNESS},
            {"auto_exposure", AUTO_EXPOSURE},
            {"shutter_speed", SHUTTER},
            {"gain_analog", GAIN},
            {"saturation", SATURATION},
            {"hue", HUE},
};

// Map of properties whose effective value is those of members "valueA" and "valueB", ints
map<string, PropertyType> CameraDriver::ints_property_map = {
            {"awb_gains", WHITE_BALANCE},
            {"sharpness", SHARPNESS},
};

bool CameraDriver::set_by_key(const string &prop_name, double value)
{
    PropertyType prop = floats_property_map[prop_name];
    Property prop_spec;
    bool automatic = (value < 0);

    prop_spec.type = prop;
    prop_spec.absControl = true;
    prop_spec.autoManualMode = automatic;
    //TODO Correct handling of automatic exposure by adding a boolean enable flag
    // to the autoexposure property
    if (!automatic) {
        prop_spec.absValue = value;
    }

    if ( camera->SetProperty(&prop_spec) != PGRERROR_OK ) {
        cerr << "Failed to set property " << prop_name << " to value: " << value << endl;
        return false;
    }

    return true;
}

bool CameraDriver::set_by_key(const string &prop_name, int valueA, int valueB)
{
    PropertyType prop = ints_property_map[prop_name];
    Property prop_spec;
    bool automatic = (valueA == -1 || valueB == -1);

    prop_spec.type = prop;
    prop_spec.autoManualMode = automatic;
    prop_spec.onOff = true;
    prop_spec.absControl = false;

    switch(prop)
    {
        case WHITE_BALANCE:
            if (!automatic) {
                prop_spec.valueA = valueA;
                prop_spec.valueB = valueB;
            }
            break;

        case SHARPNESS:
            if (!automatic) {
                prop_spec.valueA = valueA;
            }

        default:
            break;
    }

    if ( camera->SetProperty(&prop_spec) != PGRERROR_OK ) {
        cerr << "Failed to set property " << prop_name
             << " to valueA: " << valueA << " and valueB: " << valueB << endl;
        return false;
    }

    return true;
}

bool CameraDriver::is_property_float(string &prop_name)
{
    return (floats_property_map.find(prop_name) != floats_property_map.end());
}

void PrintError( Error error )
{
    error.PrintErrorTrace();
}

bool CameraDriver::get_image(Image *pImage)
{
    bool retcode = false;

    if (capturing) {
        retcode = (camera->RetrieveBuffer(pImage) == PGRERROR_OK);
    }

    return retcode;
}
