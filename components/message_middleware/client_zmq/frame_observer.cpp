#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <zmq.hpp>

#include "utils.hpp"

using namespace std;
using namespace cv;

DebugCerr debug_cerr;

int main(int argc, char **argv)
{
    int rows = 2048, cols = 2448;
    Mat frame(rows, cols, CV_8UC3);
    Mat resized(rows/2, cols/2, CV_8UC3);

    zmq::context_t context(1);
    zmq::socket_t subscriber(context, ZMQ_SUB);
    subscriber.connect(argv[1]);
    subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    debug_cerr << "Subscribed to: " << argv[1] << endl;

    while(1)
    {
        zmq::message_t message;
        subscriber.recv(&message);

        memcpy(frame.data, message.data(), message.size());
        resize(frame, resized, resized.size(), 0, 0, INTER_LINEAR);
        imshow("Camera", resized);
        //debug_cerr << "Frame painted" << endl << endl;

        char k = waitKey(10);
        if (k == 'q') break;
    }

    return 0;
}
