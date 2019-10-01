
#include <AccelStepper.h>
#include <Bounce2.h>
#include <EdgeDetector.h>
#include <PWMHigh.h>

#include "due_pins.h"
#include "MotorData.h"
#include "PulsedPin.h"

//#define TWO_POSEV_AXES
#include "PositionEvents.h"

#define NUM_MOTORS  5
AccelStepper mX(AccelStepper::DRIVER, X_STEP_PIN, X_DIR_PIN);
AccelStepper mY(AccelStepper::DRIVER, Y_STEP_PIN, Y_DIR_PIN);
AccelStepper mZ(AccelStepper::DRIVER, Z_STEP_PIN, Z_DIR_PIN);
AccelStepper mP(AccelStepper::DRIVER, E0_STEP_PIN, E0_DIR_PIN);
AccelStepper mA(AccelStepper::DRIVER, E1_STEP_PIN, E1_DIR_PIN);

MotorDataMap motors_map[NUM_MOTORS] = {
    {0x10, &mX, X_ENABLE_PIN},
    {0x20, &mY, Y_ENABLE_PIN},
    {0x30, &mZ, Z_ENABLE_PIN},
    {0x40, &mP, E0_ENABLE_PIN},
    {0x50, &mA, E1_ENABLE_PIN}
};

PulsedPin posev_fastaxis_pin(CAM_TRIGGER_PIN, true);
PulsedPin posev_flag_pin(CAM_FLAG_PIN, true);
Bounce posev_fastaxis_strobe = Bounce();

#ifdef TWO_POSEV_AXES
PositionEvents posev_fastaxis(&mA, &mP, &posev_fastaxis_pin, &posev_fastaxis_strobe, &posev_flag_pin);
int posev_index = 3;
#else
PositionEvents posev_fastaxis(&mX, &posev_fastaxis_pin, &posev_fastaxis_strobe, &posev_flag_pin);
int posev_index = 0;
#endif

/* Variables for UART communications */
#define COMM_BUFF_LENGTH    (POSEV_BUFF_LENGTH + 8)
uint8_t command[COMM_BUFF_LENGTH], uart_send[8];
int uart_index, uart_payload_len, uart_n;
bool motors_enable;

void watchdogSetup(void) {}
bool refresh_watchdog;

void manage_motors()
{
    bool done = true;
    static bool last_done = true;
    long position_done;
    AccelStepper *motor;

    posev_fastaxis_pin.update();
    posev_flag_pin.update();

    for(int i=0; i<NUM_MOTORS; i++) {
        motor = motors_map[i].motor;

        if (i==posev_index) {
            posev_fastaxis_strobe.update();
            posev_fastaxis.run(micros());
        }

        motor->run();
        done &= motor->done();

        if (motor->runFinished() || motor->hit_endstop()) {
            position_done = motor->currentPosition();
            uart_send[0] = 'm';
            uart_send[1] = i;
            uart_send[2] = position_done & 0xFF;
            uart_send[3] = (position_done >> 8)  & 0xFF;
            uart_send[4] = (position_done >> 16) & 0xFF;
            uart_send[5] = (position_done >> 24) & 0xFF;
            SerialUSB.write(uart_send, 6);
        }
    }

    if (!last_done && done) {
        SerialUSB.write("do");
    }
    last_done = done;
}

bool manage_uart(void)
{
	while (SerialUSB.available() > 0) {
		command[uart_index] = SerialUSB.read();

		if (uart_index == 0) {
			uint8_t cmd = command[0];

            if ((cmd & 0xF0) != 0) {
                MotorRegisterOffset motor_regs = (MotorRegisterOffset) (command[0] & 0x0F);
                switch(motor_regs)
                {
                    case HOLD:
                    case DIR_INVERT:
                        uart_payload_len = 1;
                        break;

                    case POS_CURRENT:
                    case POS_TARGET:
                        uart_payload_len = 4;
                        break;

                    default:
                        uart_payload_len = 2;
                }
            }
            else {
                GeneralRegisterOffset motor_regs = (GeneralRegisterOffset) (command[0] & 0x0F);
                switch(motor_regs)
                {
                    case SEQUENCE_STEPS:
                    case SEQUENCE2_STEPS:
                        uart_payload_len = (1 + POSEV_BUFF_LENGTH);
                        break;

                    case CAM_WAIT_TIME:
                        uart_payload_len = 2;
                        break;

                    default:
                        uart_payload_len = 1;
                }
            }
            // Because of the leading register index byte
            ++uart_payload_len;
		}

		if (++uart_index == uart_payload_len) {
			uart_index = 0;
			return true;
		}
	}

	return false;
}

void execute_command()
{
    bool flag;
    unsigned int i;
    uint16_t uword;
    int sword, full_int;

    AccelStepper *motor;
    unsigned int base_addr;

    flag = (bool)command[1];
    //python-smbus sends words little-endian
    uword = command[1] | (command[2] << 8);
    sword = (int16_t) uword;
    full_int = command[1] | (command[2] << 8) | (command[3] << 16 | (command[4] << 24));

    base_addr = (command[0] & 0xF0);
    if (base_addr != 0) {
        if (motors_enable) {
            // Determine which motor is being addressed
            for (i=0; i < NUM_MOTORS; i++) {
                if (base_addr == motors_map[i].base_address) {
                    motor = motors_map[i].motor;
                    break;
                }
            }
            if (i == NUM_MOTORS) {
                SerialUSB.write("nm");
                return;
            }

            // Determine which property is being written to
            MotorRegisterOffset motor_regs = (MotorRegisterOffset) (command[0] & 0x0F);
            switch(motor_regs)
            {
                case HOLD:
                    flag = (bool)command[1];
                    digitalWrite(motors_map[i].enable_pin, flag ? LOW : HIGH);
                    break;

                case STEP:
                    if (uword == 0) {
                        motor->stop();
                    }
                    else{
                        motor->move(sword);
                    }
                    break;

                case ACCEL:
                    motor->setAcceleration(uword);
                    break;

                case MAX_SPEED:
                    motor->setMaxSpeed(uword);
                    break;

                case POS_CURRENT:
                    motor->setCurrentPosition(full_int);
                    break;

                case POS_TARGET:
                    motor->moveTo(full_int);
                    break;

                case CNT_SPEED:
                    motor->setSpeed(sword);
                    break;

                case BACKLASH:
                    motor->set_backlash(uword);
                    break;

                case DIR_INVERT:
                    flag = (bool)command[1];
                    motors_map[i].motor->setPinsInverted(flag, false, false);
                    break;

                default:
                    SerialUSB.write("no");
                    return;
            }
        }
    }
    else {
        GeneralRegisterOffset registers = (GeneralRegisterOffset) (command[0] & 0x0F);
        switch(registers)
        {
            case MOTORS_STATUS:
                motors_enable = flag;
                break;

            case LED_STATE:
                digitalWrite(LED_PIN, flag? HIGH : LOW);
                break;
                
            case LED2_STATE:
                digitalWrite(LED2_PIN, flag? HIGH : LOW);
                break;

            case LED_FAN_DUTY:
                // Duty cycle normalized to the range [0, 255]
                analogWritef(LED_FAN_PIN, command[1], FAN_PWM_FREQUENCY);
                break;

            case CAM_FAN_DUTY:
                // Duty cycle normalized to the range [0, 255]
                analogWritef(CAM_FAN_PIN, command[1], FAN_PWM_FREQUENCY);
                break;

            case SEQUENCE_COMMAND:
                posev_fastaxis.execute(flag);
                break;

            case SEQUENCE_STEPS:
#ifdef TWO_POSEV_AXES
                posev_fastaxis.set_sequence(0, command[1], command + 2);
#else
                posev_fastaxis.set_sequence(command[1], command + 2);
#endif
                break;

#ifdef TWO_POSEV_AXES
            case SEQUENCE2_STEPS:
                posev_fastaxis.set_sequence(1, command[1], command + 2);
                break;
#endif

            case CAM_WAIT_TIME:
                posev_fastaxis.set_wait_time(uword);
                break;

            case CAM_SHUTTER_TIME:
                posev_fastaxis.set_shutter_time(command[1]);
                break;

            case GLOBAL_RESET:
                refresh_watchdog = false;
                break;

            default:
                SerialUSB.write("ng");
                return;
        }
    }
    SerialUSB.write("ok");
}

void A4988_initialize(AccelStepper *motor, int enable_pin)
{
    motor->setMinPulseWidth(25);
    //motor->setEnablePin(enable_pin);
    //motor->setPinsInverted(false, false, true);
}

unsigned long last, now;
int alive_state;

void setup()
{
    SerialUSB.begin(9600); ///////////////

    pinMode(LED_PIN, OUTPUT);
    pinMode(LED2_PIN, OUTPUT);
    pinMode(CAM_TRIGGER_PIN, OUTPUT);

    pinMode(X_STEP_PIN, OUTPUT);
    pinMode(X_DIR_PIN, OUTPUT);
    pinMode(X_ENABLE_PIN, OUTPUT);

    pinMode(Y_STEP_PIN, OUTPUT);
    pinMode(Y_DIR_PIN, OUTPUT);
    pinMode(Y_ENABLE_PIN, OUTPUT);

    pinMode(Z_STEP_PIN, OUTPUT);
    pinMode(Z_DIR_PIN, OUTPUT);
    pinMode(Z_ENABLE_PIN, OUTPUT);

    pinMode(E0_STEP_PIN, OUTPUT);
    pinMode(E0_DIR_PIN, OUTPUT);
    pinMode(E0_ENABLE_PIN, OUTPUT);

    pinMode(E1_STEP_PIN, OUTPUT);
    pinMode(E1_DIR_PIN, OUTPUT);
    pinMode(E1_ENABLE_PIN, OUTPUT);

    digitalWrite(X_ENABLE_PIN, HIGH);
    digitalWrite(Y_ENABLE_PIN, HIGH);
    digitalWrite(Z_ENABLE_PIN, HIGH);
    digitalWrite(E0_ENABLE_PIN, HIGH);
    digitalWrite(E1_ENABLE_PIN, HIGH);

    for (int i=0; i < NUM_MOTORS; i++) {
        A4988_initialize(motors_map[i].motor, motors_map[i].enable_pin);
    }

    mX.configure_endstops(X_MIN_PIN, X_MAX_PIN, 0);
    mY.configure_endstops(Y_MIN_PIN, Y_MAX_PIN, 0);
    mZ.configure_endstops(Z_MIN_PIN, Z_MAX_PIN, 0);
    mP.configure_endstops(E0_MIN_PIN, -1, 0);
    mA.configure_endstops(E1_MIN_PIN, -1, 0);

    digitalWrite(LED_PIN, LOW);
    digitalWrite(LED2_PIN, LOW);
    digitalWrite(CAM_TRIGGER_PIN, LOW);

    analogWritef(LED_FAN_PIN, 0, FAN_PWM_FREQUENCY);   // Pin, Dutycycle, PWM freq
    analogWritef(CAM_FAN_PIN, 0, FAN_PWM_FREQUENCY);

    pinMode(CAM_STROBE_PIN, INPUT);
    posev_fastaxis_strobe.attach(CAM_STROBE_PIN);
    posev_fastaxis_strobe.interval(1);

    while (!SerialUSB) ;

    uart_index = 0;
    SerialUSB.write("st");

    motors_enable = false;

    watchdogEnable(1000);
    refresh_watchdog = true;

    last = 0;
    alive_state = LOW;
    pinMode(ALIVE_PIN, OUTPUT);
}

void loop()
{
    manage_motors();

    if (manage_uart()) {
        execute_command();
    }

    if (refresh_watchdog) {
        watchdogReset();
    }

    now = millis();
    if (now - last >= 500) {
        last = now;
        digitalWrite(ALIVE_PIN, alive_state);
        alive_state = 1 & ~alive_state;
    }
 
}
