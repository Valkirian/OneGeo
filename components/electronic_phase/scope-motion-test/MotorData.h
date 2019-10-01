#ifndef __MOTORDATA_H__
#define __MOTORDATA_H__

#include <AccelStepper.h>

enum GeneralRegisterOffset: uint8_t {
    MOTORS_STATUS = 0x01,
    LED_STATE = 0x02,
    LED_FAN_DUTY = 0x03,
    CAM_FAN_DUTY = 0x04,
    CAM_SHUTTER_TIME = 0x07,
    CAM_WAIT_TIME = 0x09,
    SEQUENCE_COMMAND = 0x05,
    SEQUENCE_STEPS = 0x06,
    SEQUENCE2_STEPS = 0x08,
    GLOBAL_RESET = 0x0A,
    LED2_STATE = 0x0C
};

enum MotorRegisterOffset: uint8_t {
    STEP = 0x02,
    HOLD = 0x05,
    ACCEL = 0x06,
    MAX_SPEED = 0x07,
    POS_CURRENT = 0x08,
    POS_TARGET = 0x09,
    CNT_SPEED = 0x0A,
    BACKLASH = 0x0B,
    END_SPEED = 0x0D,
    DIR_INVERT = 0x0E
};

struct MotorDataMap {
    uint8_t base_address;
    AccelStepper *motor;
    int enable_pin;
};

#endif //__MOTORDATA_H__
