#include "PositionEvents.h"

#ifdef TWO_POSEV_AXES
PositionEvents::PositionEvents(AccelStepper *mA, AccelStepper *mB, PulsedPin *pin, Bounce *shutter_p, PulsedPin *pin_f)
{
  motorA = mA;
  motorB = mB;
#else
PositionEvents::PositionEvents(AccelStepper *m, PulsedPin *pin, Bounce *shutter_p, PulsedPin *pin_f)
{
  motor = m;
#endif
  pin_event = pin;
  strobe_pin = shutter_p;
  pin_flag = pin_f;

  do_run = false;
  wait_time = 50000;
  shutter_time = 2000;
  sweep_state = Idle;
}

#ifdef TWO_POSEV_AXES
void PositionEvents::set_sequence(uint8_t list_id, uint8_t count, uint8_t *buffer)
#else
void PositionEvents::set_sequence(uint8_t count, uint8_t *buffer)
#endif
{
  do_run = false;

  if (count == 0) {
    return;
  }

  uint16_t uword;
  long *list_p;
#ifdef TWO_POSEV_AXES
  list_p = (list_id == 0) ? listA : listB;
#else
  list_p = list;
#endif
  for (int i = 0; i < count; i++) {
    uword = buffer[i * 2] | (buffer[1 + i * 2] << 8);
    list_p[i] = uword;
  }
  last_index = count - 1;
}

void PositionEvents::run(unsigned long now)
{
  SweepState next_sweep_state = sweep_state;

  switch (sweep_state)
  {
    case Idle:
      if (do_run) {
        index = 0;

        if (action_at_target()) {
          next_sweep_state = PreShootWaiting;
          stop_time = add_delta_overflow(now, wait_time);
        }
        else {
          next_sweep_state = Transiting;
#ifdef TWO_POSEV_AXES
          motorA->moveTo(listA[index]);
          motorB->moveTo(listB[index]);
#else
          motor->moveTo(list[index]);
#endif
        }
        pin_flag->pulse(1000);
      }
      break;

    case PreShootWaiting:
      if (now >= stop_time) {
        next_sweep_state = Shooting;
        pin_event->pulse(1000);
        trigger_time = add_delta_overflow(now, shutter_time);
      }
      break;

    case Shooting:
      //////////
//      digitalWrite(LED_PIN, HIGH);
      if ((now >= trigger_time) || strobe_pin->fell()) {
        if (index <= last_index) {
          next_sweep_state = Transiting;
#ifdef TWO_POSEV_AXES
          motorA->moveTo(listA[index]);
          motorB->moveTo(listB[index]);
#else
          motor->moveTo(list[index]);
#endif
        }
        else {
          next_sweep_state = Idle;
          do_run = false;
          pin_flag->pulse(1000);
        }
      } //////////
    //  digitalWrite(LED_PIN,LOW);
      break;

    case Transiting:
      if (action_at_target()) {
        next_sweep_state = PreShootWaiting;
        stop_time = add_delta_overflow(now, wait_time);
      }
      break;
  }
  sweep_state = next_sweep_state;
}

bool PositionEvents::action_at_target()
{
  bool at_target;

#ifdef TWO_POSEV_AXES
  at_target = ( (listA[index] == motorA->currentPosition()) &&
                (listB[index] == motorB->currentPosition()) );
#else
  at_target = (list[index] == motor->currentPosition());
#endif

  if (at_target) {
    ++index;
    return true;
  }
  return false;
}

void PositionEvents::set_wait_time(const uint16_t wait_time_ms)
{
  wait_time = 1000 * (unsigned long) wait_time_ms;
}

void PositionEvents::set_shutter_time(const uint8_t shutter_time_ms)
{
  shutter_time = 1000 * (unsigned long) shutter_time_ms;
}

void PositionEvents::execute(bool run)
{
  do_run = run;
}

unsigned long add_delta_overflow(const unsigned long &now, const unsigned long &step)
{
  unsigned long delta;

  delta = UINT32_MAX - now;
  if (delta < step) {
    return (step - delta);
  }
  else {
    return now + step;
  }
}
