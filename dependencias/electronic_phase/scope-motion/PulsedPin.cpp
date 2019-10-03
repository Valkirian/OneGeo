#include "PulsedPin.h"

PulsedPin::PulsedPin(int p, bool active_hi)
{
    pin = p;
    active_high = active_hi;
    state = false;

    pinMode(pin, OUTPUT);
    digitalWrite(pin, active_high ? LOW : HIGH);
}

void PulsedPin::pulse(unsigned long pulse_time)
{
    digitalWrite(pin, active_high ? HIGH : LOW);
    trigger_time = micros();
    active_time = pulse_time;
    state = true;
}

void PulsedPin::update()
{
    if (state && (micros() - trigger_time >= active_time)) {
        state = false;
        digitalWrite(pin, active_high ? LOW : HIGH);
    }
}
