#ifndef __POSEVENTS_H__
#define __POSEVENTS_H__

#include <Arduino.h>

#include <AccelStepper.h>
#include <Bounce2.h>
#include "PulsedPin.h"

#define TWO_POSEV_AXES
#define POSEV_LIST_LENGTH   32
#define POSEV_BUFF_LENGTH   (2*POSEV_LIST_LENGTH)


class PositionEvents
{
    enum SweepState : uint8_t {
        Idle,
        PreShootWaiting,
        Shooting,
        Transiting
    };

#ifdef TWO_POSEV_AXES
    AccelStepper  *motorA;
    AccelStepper  *motorB;
    long          listA[POSEV_LIST_LENGTH];
    long          listB[POSEV_LIST_LENGTH];
#else
    AccelStepper  *motor;
    long          list[POSEV_LIST_LENGTH];
#endif

    Bounce        *strobe_pin;
    PulsedPin     *pin_event;
    PulsedPin     *pin_flag;
    SweepState    sweep_state;
    uint8_t       index;
    uint8_t       last_index;
    bool          do_run;
    // These times are in microseconds (i.e. compared against micros())
    unsigned long trigger_time;
    unsigned long stop_time;
    unsigned long shutter_time;
    unsigned long wait_time;

    public:

#ifdef TWO_POSEV_AXES
        PositionEvents(AccelStepper*, AccelStepper*, PulsedPin*, Bounce*, PulsedPin*);
        void set_sequence(uint8_t, uint8_t, uint8_t*);
#else
        PositionEvents(AccelStepper*, PulsedPin*, Bounce*, PulsedPin*);
        void set_sequence(uint8_t, uint8_t*);
#endif

        void set_wait_time(uint16_t);
        void set_shutter_time(uint8_t);
        void run(unsigned long);
        void execute(bool);

    private:

        bool action_at_target(void);
};

unsigned long add_delta_overflow(const unsigned long&, const unsigned long&);

#endif //__POSEVENTS_H__
