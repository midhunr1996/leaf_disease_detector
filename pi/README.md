# Raspberry Pi 5 deployment

Wiring the L298N motor driver and running the tea leaf detector on the Pi.

---

## 1. Wiring

### Parts

| Part | Notes |
|---|---|
| Raspberry Pi 5 | Its own 5 V / 5 A supply |
| L298N dual H-bridge module | The common red breakout board |
| DC gear motor | 12 V, driving the belt |
| Motor supply | 7-12 V, rated for the motor's stall current |
| Logitech C922 | USB |

### Connections

**Driver side:**

| L298N | Wire to | Pi 5 header | Purpose |
|---|---|---|---|
| **ENA** | GPIO12 | pin **32** | Speed (PWM duty cycle) |
| **IN1** | GPIO23 | pin **16** | Direction bit 1 |
| **IN2** | GPIO24 | pin **18** | Direction bit 2 |
| **GND** | Pi ground | pin **34** | Common reference, mandatory |
| **+12V** | Supply + | none | Motor power |
| **GND** | Supply - | (same as pin 34) | Shared ground |
| **+5V** | **nothing** | none | Regulator output, see warning |
| OUT1 | Motor **M1** | none | |
| OUT2 | Motor **M2** | none | |

**Motor side (6-wire encoder motor).** The board silkscreen reads, top to
bottom: `M1, GND, C2, C1, VCC, M2`.

| Motor pad | Goes to | Pi 5 header |
|---|---|---|
| **M1** | L298N OUT1 | none |
| **M2** | L298N OUT2 | none |
| **VCC** | Pi **3.3 V** | pin **1** or **17** |
| **GND** | Pi ground | pin **30** |
| **C1** | GPIO5, channel A | pin **29** |
| **C2** | GPIO6, channel B | pin **31** |

> ### Power the encoder from 3.3 V, never 5 V
>
> The encoder's outputs swing to whatever its supply is. On 5 V, C1 and C2
> will present 5 V to the Pi, and **the Pi 5's GPIO pins are not 5 V
> tolerant** — you will damage them. Take VCC from a 3.3 V pin so the outputs
> come back at 3.3 V.

> **Do not trust the wire colours.** On these modules red is often encoder VCC
> rather than motor positive, and vendors differ. Buzz each wire to its
> labelled pad with a multimeter: M1 and M2 read a couple of ohms through the
> windings, VCC and GND do not.

```
  Raspberry Pi 5                  L298N                 Motor
  +--------------+          +--------------+         +--------+
  | GPIO12 pin32 |--PWM---->| ENA          |         |        |
  | GPIO23 pin16 |--------->| IN1     OUT1 |-------->|   M    |
  | GPIO24 pin18 |--------->| IN2     OUT2 |-------->|        |
  | GND    pin34 |----------| GND          |         +--------+
  +--------------+     |    | +12V         |<---- 12 V supply +
                       +----| GND          |<---- 12 V supply -
                            | +5V   leave  |
                            +--------------+
      the Pi keeps its own 5 V supply; grounds are common
```

### Five things that decide whether this works

1. **Join the grounds.** The Pi and the motor supply must share a 0 V
   reference, or the driver has nothing to compare the 3.3 V logic against and
   the inputs float. This is the most common wiring mistake.
2. **3.3 V logic is fine.** The L298 datasheet gives a minimum high-level input
   of 2.3 V, so the Pi's 3.3 V reads as a solid logic one. No level shifter is
   needed.
3. **Never connect L298N `+5V` to the Pi.** That pin is a small onboard
   regulator good for a few hundred milliamps. A Pi 5 draws several amps and
   would brown out. If your motor supply is above 12 V, pull the 5 V jumper and
   feed the logic side separately.
4. **Remove the ENA jumper.** Modules ship with ENA and ENB jumpered to 5 V,
   which forces full speed and ignores the Pi entirely.
5. **Expect a 2 V drop.** The L298N is bipolar, not MOSFET, so 12 V in gives
   roughly 10 V at the motor and the rest becomes heat. Fit a heatsink above
   about 1 A continuous.

### Control logic

| IN1 | IN2 | ENA | Motor |
|:---:|:---:|:---:|---|
| 0 | 0 | any | Coast to stop |
| 1 | 0 | PWM | Forward at duty |
| 0 | 1 | PWM | Reverse at duty |
| 1 | 1 | any | Brake, windings shorted |
| any | any | 0 | Off |

### Safety

An emergency stop in software is not an emergency stop. Put a physical switch
or contactor in the **motor supply line** so the belt can be killed even if the
Pi has hung.

---

## 2. Software

Raspberry Pi OS (64-bit), Bookworm or later.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-libgpiod \
                    libgl1 libglib2.0-0 v4l-utils

mkdir -p ~/leaf && cd ~/leaf
# copy the pi/ folder and the pi5_export/ folder here

python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime opencv-python-headless flask gpiozero lgpio numpy
```

### Two traps

**`RPi.GPIO` does not work on the Pi 5.** The Pi 5 moved the GPIO onto the RP1
controller chip, so libraries that map `/dev/mem` directly cannot reach the
pins. Use `gpiozero` on the `lgpio` backend, which is what this code does, or
install `rpi-lgpio` as a drop-in replacement for legacy code. Do not install
`RPi.GPIO` itself: if it is present, gpiozero may select it and fail at runtime.

**Bookworm marks the system Python as externally managed.** A plain
`pip install` fails by design. Hence the venv above. `--system-site-packages`
keeps the apt-installed `python3-libgpiod` visible inside it.

### Check the camera

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext | head -30
```

---

## 3. Run it

```bash
cd ~/leaf/pi
source ../venv/bin/activate

python motor.py                  # bench test: forward, reverse, brake
python server.py --no-motor      # detector only, no GPIO
python server.py                 # full system
```

Then open `http://<pi-address>:8000/`.

| Flag | Default | |
|---|---|---|
| `--model` | `../pi5_export/YOLO26s_640_fp32.onnx` | |
| `--camera` | `0` | `/dev/videoN` index |
| `--width` `--height` `--fps` | `1280 720 30` | |
| `--port` | `8000` | |
| `--no-motor` | off | Skip GPIO entirely |

### Which model file

> **The int8 export is broken. Do not deploy it.**
>
> Tested on five real tea photographs against the fp32 twin:
>
> | Model | Boxes found | Max class score |
> |---|---|---|
> | `YOLO26s_640_fp32.onnx` | 22 | 0.81 - 0.92 |
> | `YOLO26s_640_int8.onnx` | **0** | **0.0000** |
>
> Quantisation destroyed the classification head: every candidate returns a
> score of exactly zero, so nothing ever clears the confidence threshold. The
> box head survived, with coordinates within about 1 % of fp32, which is why
> the file loads and runs and looks healthy. It simply never detects anything.
>
> This is not fixable by lowering `conf`. A score of 0.0 is below any
> threshold.

| File | Size | Use |
|---|---|---|
| `YOLO26s_640_fp32.onnx` | 38 MB | **Default. The only working file.** |
| `YOLO26s_640_int8.onnx` | 10.3 MB | Broken, see above |

**To re-export int8 properly**, quantisation needs calibration data: a few
hundred representative images run through the model so the tool can measure the
real range of each tensor. Dynamic quantisation with no calibration set is the
usual cause of a collapsed classification head. Re-export, then re-run the
comparison above before trusting it.

fp32 measures about 111 ms median per image on a desktop CPU, including
letterbox and postprocess. Expect the Pi to be slower and measure it there.

---

## 4. Start on boot

`/etc/systemd/system/leafdet.service`, adjusting the paths and user:

```ini
[Unit]
Description=Tea leaf inspection
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/leaf/pi
ExecStart=/home/pi/leaf/venv/bin/python /home/pi/leaf/pi/server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now leafdet
journalctl -u leafdet -f
```

---

## 5. Why inference runs on the Pi, not the browser

The browser version in the project root (`index.html` and `app.js`) runs the
model in whoever's browser opens the page. That is convenient for a demo and
wrong for machine control:

- The Pi owns the motor, but the **browser** would be deciding when to stop it.
- Close the tab, or let a phone lock its screen, and **the belt keeps running
  with nothing watching it**.

`server.py` runs detection and the auto-stop in the same process that owns the
GPIO pins, so the stop fires whether or not anyone has the page open. The
browser becomes a viewer.

Keep the browser version for reviewing images and for demonstrating the model
away from the hardware.

---

## 6. API

| Method | Path | Body and notes |
|---|---|---|
| GET | `/video` | MJPEG stream with boxes drawn |
| GET | `/api/status` | Motor, detector, counts, `defect_held_s` |
| POST | `/api/motor` | `{"action":"start\|stop\|brake\|speed\|estop\|reset_estop\|acknowledge","duty":0.5}` |
| POST | `/api/settings` | `{"conf":0.35,"iou":0.45,"hold_s":5.0,"recording":true}` |
| GET | `/api/records` | `?limit=200` |
| GET | `/api/records.csv` | Full export |

Records append to `records.jsonl`, one JSON object per line:

```json
{"id":1,"ts":"2026-09-13T14:35:02","cls":"defect","conf":0.87,
 "box":[12,44,180,220],"speed":0.1,"event":"","device":"pi5-line-01"}
```

`event` is empty for ordinary detections and `"auto-stop after 5.1s"` for the
record written when a held defect stops the motor.

---

## 7. Calibrate the encoder

The encoder measures belt speed directly, so records carry what the belt did
rather than what was asked of it. Two constants in `motor.py` must be set for
your hardware first, or the measured speed will be confidently wrong.

**`COUNTS_PER_REV`** — counts per revolution of the **output** shaft:

```
COUNTS_PER_REV = PPR × 4 × gear_ratio
```

The ×4 is because a quadrature decoder sees four edges per pulse. An 11 PPR
encoder behind a 34:1 gearbox gives `11 × 4 × 34 = 1496`. Both numbers are on
the motor's datasheet. If you do not have it, measure instead: mark the output
shaft, turn it by hand exactly ten revolutions, and read `e.counts`.

**`PULLEY_DIAMETER_M`** — the drive pulley diameter in metres. Belt speed is
`π × D` per output revolution. Measure across the belt contact surface, not the
flange.

### Verify

```bash
python motor.py            # should report non-zero rev/s and counts
python motor.py calibrate  # duty against measured speed, 0.2 to 1.0
```

Cross-check once against a ruler and a stopwatch: mark the belt, run 10 s,
measure the distance. If that disagrees with the encoder, one of the two
constants is wrong.

Paste the calibration table into the thesis — it fills the duty-against-speed
placeholder in Section 5.6.

### If counts stay at zero

- C1/C2 not actually on GPIO5/GPIO6
- Encoder VCC not connected, or on 5 V and the input already damaged
- Encoder ground not shared with the Pi

### A note on pulse rate

`gpiozero.RotaryEncoder` decodes in Python, which is fine at moderate speeds
but will start dropping counts in the low kHz. At 1496 counts/rev and 100 rpm
output you are at about 2.5 kHz, which is near the practical limit. Dropped
counts show up as a speed reading that is too low and drifts. If you see that,
either sample only channel A for speed, or move the counting to a hardware
peripheral.
