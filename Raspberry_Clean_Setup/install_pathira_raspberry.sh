#!/usr/bin/env bash
set -e

APP_DIR="$HOME/pathira_raspberry"
mkdir -p "$APP_DIR"
cp ./pathira_raspberry_client.py "$APP_DIR/"
cp ./pathira_status_leds.py "$APP_DIR/"
chmod +x "$APP_DIR/pathira_raspberry_client.py" "$APP_DIR/pathira_status_leds.py"

sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-requests python3-gpiozero libcamera-apps avahi-daemon network-manager

sudo raspi-config nonint do_camera 0 || true
sudo raspi-config nonint do_i2c 0 || true
sudo raspi-config nonint do_wifi_country EG || true
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

sudo tee /etc/systemd/system/pathira-raspberry-camera.service >/dev/null <<SERVICE
[Unit]
Description=Pathira Raspberry Camera sender to Android phone
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 $APP_DIR/pathira_raspberry_client.py --phone-host auto --phone-port 8080 --fps 30 --ai-fps 3 --jpeg-quality 70 --width 854 --height 480 --color-mode swap-rb
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable --now pathira-raspberry-camera.service
sudo systemctl enable --now pathira-status-leds.service

echo "Installed Pathira Raspberry camera service."
echo "Start now: sudo systemctl start pathira-raspberry-camera.service"
echo "Logs: sudo journalctl -u pathira-raspberry-camera.service -f"
echo "LED logs: sudo journalctl -u pathira-status-leds.service -f"
