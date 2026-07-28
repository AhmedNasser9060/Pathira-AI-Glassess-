#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/pathira_raspberry"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$APP_DIR"
install -m 0755 "$SCRIPT_DIR/pathira_status_leds.py" "$APP_DIR/pathira_status_leds.py"

sudo apt-get update
sudo apt-get install -y python3-gpiozero network-manager
sudo usermod -a -G gpio "$USER" || true

ACTIVE_PROFILE="$(nmcli -g GENERAL.CONNECTION device show wlan0 2>/dev/null || true)"
if [ -n "$ACTIVE_PROFILE" ] && [ "$ACTIVE_PROFILE" != "--" ]; then
  sudo nmcli connection modify "$ACTIVE_PROFILE" \
    connection.autoconnect yes \
    connection.autoconnect-priority 100 \
    802-11-wireless-security.key-mgmt wpa-psk \
    802-11-wireless-security.psk Ahmed100
fi

sudo tee /etc/systemd/system/pathira-status-leds.service >/dev/null <<SERVICE
[Unit]
Description=Pathira hotspot status LEDs
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
Environment=PATHIRA_HOTSPOT_SSID=Ahmed
Environment=PATHIRA_RED_LED_GPIO=12
Environment=PATHIRA_GREEN_LED_GPIO=16
ExecStart=/usr/bin/python3 $APP_DIR/pathira_status_leds.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable --now pathira-status-leds.service
sleep 2
sudo systemctl --no-pager --full status pathira-status-leds.service

echo
echo "Pathira status LEDs installed."
echo "Red: BCM GPIO 12 (physical pin 32)"
echo "Green: BCM GPIO 16 (physical pin 36)"
echo "SSID: Ahmed"
