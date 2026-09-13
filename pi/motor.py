"""L298N motor control for the Raspberry Pi 5.

Wiring (see thesis Table 5.x and Figure 5.x):

    L298N ENA  -> GPIO12  (header pin 32)   speed, as a PWM duty cycle
    L298N IN1  -> GPIO23  (header pin 16)   direction bit 1
    L298N IN2  -> GPIO24  (header pin 18)   direction bit 2
    L298N GND  -> Pi GND  (header pin 34)   common reference, mandatory
    L298N +12V -> motor supply positive
    L298N GND  -> motor supply negative     (same ground as the Pi)
    L298N +5V  -> NOTHING. It is a small regulator output and must never be
                  connected to the Pi, which draws far more than it can supply.

Remove the ENA jumper on the module, otherwise the enable pin is tied high and
the speed setting is ignored.

Library note: RPi.GPIO does not work on the Pi 5. The Pi 5 moved the pins onto
the RP1 controller, so libraries that map /dev/mem directly cannot reach them.
gpiozero on the lgpio backend is the supported route and is what is used here.
"""

import math
import threading
import time

from gpiozero import DigitalOutputDevice, PWMOutputDevice, RotaryEncoder

ENA_PIN = 12
IN1_PIN = 23
IN2_PIN = 24

# Quadrature encoder channels. GPIO5 and GPIO6 are free and sit next to a
# ground pin, which keeps the two signal wires short.
C1_PIN = 5          # header pin 29, encoder channel A
C2_PIN = 6          # header pin 31, encoder channel B

# --- calibration, both MUST be measured for this motor and belt -------------
# Counts per revolution of the OUTPUT shaft, after the gearbox:
#     COUNTS_PER_REV = pulses_per_rev_of_motor * 4 * gear_ratio
# The factor of 4 is because a quadrature decoder sees four edges per pulse.
# A 11 PPR encoder behind a 34:1 gearbox gives 11 * 4 * 34 = 1496.
COUNTS_PER_REV = 1496.0

# Diameter of the drive pulley the belt runs on, in metres. Belt speed is
# pi * D per output revolution.
PULLEY_DIAMETER_M = 0.030

# Below this duty cycle a small geared motor buzzes without turning, so a
# request under it is treated as a stop rather than pretending to run.
MIN_DUTY = 0.15

# Software PWM frequency. Low enough for the bridge to switch cleanly, high
# enough to stay out of the range people find audible and irritating.
PWM_HZ = 1000


class Motor:
    """One channel of an L298N driving the conveyor belt."""

    def __init__(self, ena=ENA_PIN, in1=IN1_PIN, in2=IN2_PIN, pwm_hz=PWM_HZ):
        self._ena = PWMOutputDevice(ena, frequency=pwm_hz, initial_value=0.0)
        self._in1 = DigitalOutputDevice(in1, initial_value=False)
        self._in2 = DigitalOutputDevice(in2, initial_value=False)
        self._duty = 0.0
        self._running = False
        self._reverse = False

    @property
    def state(self):
        return {
            "running": self._running,
            "duty": round(self._duty, 3),
            "reverse": self._reverse,
        }

    def start(self, duty=0.5, reverse=False):
        """Turn the motor on at a duty cycle between 0 and 1."""
        duty = max(0.0, min(1.0, float(duty)))
        if duty < MIN_DUTY:
            self.stop()
            return self.state

        # Direction first, then enable, so the bridge never sees an enable
        # pulse while both direction inputs are still changing.
        self._reverse = bool(reverse)
        if self._reverse:
            self._in1.off()
            self._in2.on()
        else:
            self._in1.on()
            self._in2.off()

        self._ena.value = duty
        self._duty = duty
        self._running = True
        return self.state

    def set_speed(self, duty):
        """Change speed without changing direction."""
        if not self._running:
            return self.state
        return self.start(duty, self._reverse)

    def stop(self):
        """Cut drive and let the belt coast to rest."""
        self._ena.value = 0.0
        self._in1.off()
        self._in2.off()
        self._duty = 0.0
        self._running = False
        return self.state

    def brake(self):
        """Short the windings so the belt stops sharply.

        Used for the emergency stop and the automatic defect stop, where the
        belt should not carry on drifting past the inspection point.
        """
        self._in1.on()
        self._in2.on()
        self._ena.value = 1.0
        self._duty = 0.0
        self._running = False
        return self.state

    def close(self):
        self.stop()
        for dev in (self._ena, self._in1, self._in2):
            dev.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Encoder:
    """Quadrature encoder on the motor's output shaft.

    Turns edge counts into a measured belt speed, so the records carry what the
    belt actually did rather than what the operator asked for. That distinction
    matters: the motion-blur limit in the feasibility analysis is a function of
    real speed, and a duty cycle is not a speed.

    IMPORTANT: power the encoder from 3.3 V. Its outputs swing to whatever its
    supply is, and the Pi 5's GPIO pins are not 5 V tolerant.
    """

    def __init__(self, a=C1_PIN, b=C2_PIN,
                 counts_per_rev=COUNTS_PER_REV,
                 pulley_d=PULLEY_DIAMETER_M,
                 window_s=0.5):
        # max_steps=0 lets the count run unbounded instead of wrapping.
        self._enc = RotaryEncoder(a, b, max_steps=0)
        self.counts_per_rev = float(counts_per_rev)
        self.pulley_d = float(pulley_d)
        self._window_s = window_s
        self._lock = threading.Lock()
        self._rev_s = 0.0
        self._last_steps = 0
        self._last_t = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while self._running:
            time.sleep(self._window_s)
            now = time.monotonic()
            steps = self._enc.steps
            dt = now - self._last_t
            if dt > 0:
                d_counts = steps - self._last_steps
                with self._lock:
                    self._rev_s = (d_counts / self.counts_per_rev) / dt
            self._last_steps, self._last_t = steps, now

    @property
    def counts(self):
        return self._enc.steps

    @property
    def rev_per_s(self):
        with self._lock:
            return self._rev_s

    @property
    def speed_mps(self):
        """Belt speed in metres per second. Sign follows direction."""
        return self.rev_per_s * math.pi * self.pulley_d

    def reset(self):
        self._enc.steps = 0
        self._last_steps = 0

    def close(self):
        self._running = False
        self._enc.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def calibrate(motor, encoder, duties=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
              settle_s=2.0):
    """Measure belt speed against duty cycle.

    Duty is not speed, and the relationship is neither linear nor constant
    under load. Run this with the belt loaded the way it will be in service and
    paste the table into the thesis.
    """
    print("%6s %10s %12s" % ("duty", "rev/s", "m/s"))
    rows = []
    for d in duties:
        motor.start(d)
        time.sleep(settle_s)
        rev_s, mps = encoder.rev_per_s, encoder.speed_mps
        rows.append((d, rev_s, mps))
        print("%6.2f %10.3f %12.4f" % (d, rev_s, mps))
    motor.stop()
    return rows


if __name__ == "__main__":
    # Bench test. Run with the belt clear of anything that can be dragged in.
    #   python motor.py             forward, reverse, brake, reading the encoder
    #   python motor.py calibrate   duty against measured speed
    import sys

    with Motor() as m, Encoder() as e:
        if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
            calibrate(m, e)
        else:
            for duty in (0.3, 0.5, 0.8):
                m.start(duty)
                time.sleep(2)
                print("forward duty=%.1f  %.3f rev/s  %.4f m/s  %d counts"
                      % (duty, e.rev_per_s, e.speed_mps, e.counts))

            m.start(0.5, reverse=True)
            time.sleep(2)
            print("reverse         %.3f rev/s  %.4f m/s  %d counts"
                  % (e.rev_per_s, e.speed_mps, e.counts))

            m.brake()
            time.sleep(1)
            print("brake           %.3f rev/s (should be ~0)" % e.rev_per_s)
            m.stop()

        if e.counts == 0:
            print("\nEncoder read zero counts. Check that C1/C2 reach GPIO5/6,"
                  "\nand that the encoder VCC is on 3.3 V rather than 5 V.")
