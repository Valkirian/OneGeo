#ifndef __PULSEDPIN_H__
#define __PULSEDPIN_H__

#include <Arduino.h>

class PulsedPin
{
    int pin;
    bool state;
    bool active_high;
    unsigned long trigger_time;
    unsigned long active_time;

    public:

        PulsedPin(int, bool);
        void pulse(unsigned long);
        void update();
};

#endif //__PULSEDPIN_H__

