#!/usr/bin/env bash
# Deploy the tea leaf inspection system to the Raspberry Pi.
#
#   bash pi/deploy.sh                       # code + model + service
#   bash pi/deploy.sh --code-only           # skip the model (much faster)
#   PI_HOST=192.168.1.130 bash pi/deploy.sh # override the address
#
# Requires key-based SSH. Set it up once with:
#   ssh-copy-id -i ~/.ssh/id_ed25519.pub pi5@pi5.local
#
# No password is stored in this script, and none should be added to it.

set -euo pipefail

PI_USER="${PI_USER:-pi5}"
PI_HOST="${PI_HOST:-pi5.local}"
PI_DIR="${PI_DIR:-/home/${PI_USER}/leaf}"
MODEL="${MODEL:-pi5_export/YOLO26s_640_fp32.onnx}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CODE_ONLY=0
[ "${1:-}" = "--code-only" ] && CODE_ONLY=1

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

say "Checking connection to ${PI_USER}@${PI_HOST}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${PI_USER}@${PI_HOST}" \
    'echo "  connected: $(uname -m), $(python3 --version)"' || {
  echo "Could not connect without a password."
  echo "Run:  ssh-copy-id -i ~/.ssh/id_ed25519.pub ${PI_USER}@${PI_HOST}"
  exit 1
}

say "Creating ${PI_DIR}"
ssh "${PI_USER}@${PI_HOST}" "mkdir -p '${PI_DIR}/pi' '${PI_DIR}/model'"

say "Copying code"
scp -q "${HERE}/pi/motor.py" "${HERE}/pi/server.py" \
       "${HERE}/pi/requirements.txt" "${HERE}/pi/README.md" \
       "${PI_USER}@${PI_HOST}:${PI_DIR}/pi/"
if [ -f "${HERE}/pi/uploader.py" ]; then
  scp -q "${HERE}/pi/uploader.py" "${PI_USER}@${PI_HOST}:${PI_DIR}/pi/"
fi

if [ "$CODE_ONLY" -eq 0 ]; then
  say "Copying model ($(du -h "${HERE}/${MODEL}" | cut -f1)), this takes a minute"
  scp "${HERE}/${MODEL}" "${PI_USER}@${PI_HOST}:${PI_DIR}/model/model.onnx"
else
  say "Skipping model (--code-only)"
fi

say "Installing system packages and the virtual environment"
ssh "${PI_USER}@${PI_HOST}" bash -s <<REMOTE
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-libgpiod libgl1 \
                           libglib2.0-0 v4l-utils

if [ ! -d "${PI_DIR}/venv" ]; then
  python3 -m venv --system-site-packages "${PI_DIR}/venv"
fi
"${PI_DIR}/venv/bin/pip" install -q --upgrade pip
"${PI_DIR}/venv/bin/pip" install -q -r "${PI_DIR}/pi/requirements.txt"
echo "  packages installed"

# RPi.GPIO cannot reach the Pi 5's pins. If something pulled it in, gpiozero
# may select it and fail at runtime, so remove it.
if "${PI_DIR}/venv/bin/pip" show RPi.GPIO >/dev/null 2>&1; then
  echo "  removing RPi.GPIO, which does not work on the Pi 5"
  "${PI_DIR}/venv/bin/pip" uninstall -y -q RPi.GPIO
fi
REMOTE

say "Installing the service"
ssh "${PI_USER}@${PI_HOST}" bash -s <<REMOTE
set -e
sudo tee /etc/systemd/system/leafdet.service >/dev/null <<UNIT
[Unit]
Description=Tea leaf inspection
After=network-online.target

[Service]
Type=simple
User=${PI_USER}
WorkingDirectory=${PI_DIR}/pi
EnvironmentFile=-${PI_DIR}/pi/.env
ExecStart=${PI_DIR}/venv/bin/python ${PI_DIR}/pi/server.py --model ${PI_DIR}/model/model.onnx
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable leafdet >/dev/null 2>&1
sudo systemctl restart leafdet
sleep 3
if systemctl is-active --quiet leafdet; then
  echo "  service running"
else
  echo "  service failed, last log lines:"
  journalctl -u leafdet -n 20 --no-pager
fi
REMOTE

say "Checking the camera"
ssh "${PI_USER}@${PI_HOST}" \
  'v4l2-ctl --list-devices 2>/dev/null | head -6 || echo "  v4l2-ctl found nothing"'

IP=$(ssh "${PI_USER}@${PI_HOST}" "hostname -I | awk '{print \$1}'")
say "Done"
echo "  interface : http://${IP}:8000/  (or http://${PI_HOST}:8000/)"
echo "  logs      : ssh ${PI_USER}@${PI_HOST} 'journalctl -u leafdet -f'"
echo "  restart   : ssh ${PI_USER}@${PI_HOST} 'sudo systemctl restart leafdet'"
