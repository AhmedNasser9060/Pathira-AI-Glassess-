#!/usr/bin/env python3
"""Show the Pathira hotspot state using two Raspberry Pi LEDs."""

import os
import signal
import subprocess
import time

from gpiozero import LED


RED_GPIO = int(os.getenv("PATHIRA_RED_LED_GPIO", "12"))
GREEN_GPIO = int(os.getenv("PATHIRA_GREEN_LED_GPIO", "16"))
HOTSPOT_SSID = os.getenv("PATHIRA_HOTSPOT_SSID", "Ahmed")
BLINK_SECONDS = 0.45

running = True


def stop(_signum, _frame):
    global running
    running = False


def command_output(command):
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def hotspot_connected():
    connections = command_output(
        ["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi"]
    )
    expected = f"yes:{HOTSPOT_SSID}"
    if expected not in connections.splitlines():
        return False

    address = command_output(["ip", "-4", "address", "show", "wlan0"])
    return "inet " in address


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    red = LED(RED_GPIO, active_high=True, initial_value=False)
    green = LED(GREEN_GPIO, active_high=True, initial_value=False)
    red_on = False

    try:
        while running:
            if hotspot_connected():
                red.off()
                green.on()
                red_on = False
            else:
                green.off()
                red_on = not red_on
                red.on() if red_on else red.off()
            time.sleep(BLINK_SECONDS)
    finally:
        red.off()
        green.off()
        red.close()
        green.close()


if __name__ == "__main__":
    main()
