#!/usr/bin/env python3
"""Pathira Raspberry camera sender for the standalone Android APK.

This mirrors the stable Nasser PC pipeline:
- capture camera frames continuously
- upload raw frames for smooth preview on a fast loop
- upload AI frames on a slower independent loop
"""

import argparse
import ipaddress
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests
from picamera2 import Picamera2

try:
    from gpiozero import Button
except Exception:
    Button = None

DEFAULT_PORT = int(os.getenv("PATHIRA_PHONE_PORT", "8080"))
DEFAULT_HOST = os.getenv("PATHIRA_PHONE_HOST", "auto")
PTT_GPIO_PIN = 24
RAW_STREAM_ENDPOINT = "/camera/stream/raw"
ENDPOINTS = {
    "detect": "/camera/stream/object",
    "object": "/camera/stream/object",
    "obstacle": "/camera/stream/obstacle",
    "ocr": "/camera/stream/ocr",
    "scene": "/camera/stream/scene",
    "face": "/camera/stream/face",
}


def run_text(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=2, check=False).stdout.strip()
    except Exception:
        return ""


def default_gateway():
    parts = run_text(["ip", "route", "show", "default"]).split()
    if "via" in parts:
        index = parts.index("via")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def local_ipv4_and_network():
    for iface in ("wlan0", "end0", "eth0"):
        output = run_text(["ip", "-4", "addr", "show", iface])
        for token in output.replace("\n", " ").split():
            if "/" in token:
                try:
                    interface = ipaddress.ip_interface(token)
                    if not interface.ip.is_loopback:
                        return str(interface.ip), interface.network
                except Exception:
                    pass
    return None, None


def health_ok(host, port, timeout=0.55):
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200 and "pathira" in response.text.lower()
    except Exception:
        return False


def discover_phone(host, port):
    if host and host.lower() != "auto":
        return host
    candidates = []
    gateway = default_gateway()
    if gateway:
        candidates.append(gateway)
    candidates += ["192.168.43.1", "192.168.1.1", "192.168.137.1", "172.20.10.1"]
    for candidate in dict.fromkeys(candidates):
        print(f"Checking phone server at {candidate}:{port} ...")
        if health_ok(candidate, port):
            return candidate
    local_ip, network = local_ipv4_and_network()
    if network is None:
        raise RuntimeError("Could not detect WiFi network.")
    print(f"Scanning WiFi network {network} for Pathira phone server on port {port} ...")
    hosts = [str(ip) for ip in network.hosts() if str(ip) != local_ip]
    with ThreadPoolExecutor(max_workers=48) as pool:
        future_map = {pool.submit(health_ok, h, port, 0.45): h for h in hosts}
        for future in as_completed(future_map):
            if future.result():
                return future_map[future]
    raise RuntimeError("Could not find phone app server. Use --phone-host PHONE_IP from app Debug card.")


class PathiraRaspberryClient:
    def __init__(self, phone_host, port, fps, ai_fps, jpeg_quality, width, height, color_mode):
        self.base_url = f"http://{phone_host}:{port}"
        self.fps = max(1, fps)
        self.frame_delay = 1.0 / self.fps
        self.ai_fps = max(1, ai_fps)
        self.ai_frame_delay = 1.0 / self.ai_fps
        self.jpeg_quality = jpeg_quality
        self.width = width
        self.height = height
        self.color_mode = color_mode
        self.picam2 = None
        self.running = True
        self.current_mode = "detect"
        self.capture_count = 0
        self.raw_sent = 0
        self.ai_sent = 0
        self.failed = 0
        self.latest_jpeg = None
        self.latest_frame_id = 0
        self.latest_ai_frame_id = 0
        self.frame_lock = threading.Lock()
        self.control_session = self._build_session()
        self.raw_session = self._build_session()
        self.ai_session = self._build_session()
        self.ptt_button = None
        self._setup_ptt()

    def _build_session(self):
        session = requests.Session()
        session.headers.update({"Connection": "keep-alive"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        session.mount("http://", adapter)
        return session

    def _setup_ptt(self):
        if Button is None:
            return
        try:
            self.ptt_button = Button(PTT_GPIO_PIN, pull_up=True, bounce_time=0.03)
        except Exception as exc:
            print(f"GPIO {PTT_GPIO_PIN} PTT disabled: {exc}")

    def ptt_pressed(self):
        try:
            return bool(self.ptt_button and self.ptt_button.is_pressed)
        except Exception:
            return False

    def start_camera(self):
        print("Starting Picamera2...")
        self.picam2 = Picamera2()
        if not Picamera2.global_camera_info():
            raise RuntimeError("No Raspberry camera detected.")
        camera_format = "RGB888" if self.color_mode == "rgb" else "BGR888"
        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": camera_format},
            buffer_count=2,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)
        self._capture_frame()
        print("Camera is ready.")

    def _capture_frame(self):
        last_error = None
        for attempt in range(1, 4):
            try:
                return self.picam2.capture_array()
            except Exception as exc:
                last_error = exc
                print(f"Camera frame capture failed, retry {attempt}/3: {exc}")
                time.sleep(0.5)
        raise RuntimeError(f"Camera did not produce frames. Check ribbon cable/port. Last error: {last_error}")

    def encode_frame(self, frame):
        if self.color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.color_mode == "swap-rb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
        return buffer.tobytes() if ok else None

    def get_mode(self):
        try:
            response = self.control_session.get(f"{self.base_url}/camera/current-model", timeout=0.5)
            if response.status_code == 200:
                mode = str(response.json().get("model", self.current_mode)).lower()
                if mode == "raw":
                    self.current_mode = "raw"
                elif mode in ENDPOINTS:
                    self.current_mode = mode
        except Exception:
            pass
        return self.current_mode

    def send_frame_bytes(self, jpeg_bytes, endpoint, session, timeout):
        try:
            response = session.post(
                f"{self.base_url}{endpoint}",
                headers={"X-PTT-Pressed": "1" if self.ptt_pressed() else "0"},
                files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
                timeout=timeout,
            )
            return response.status_code == 200
        except Exception:
            return False

    def capture_loop(self):
        while self.running:
            started = time.time()
            frame = self._capture_frame()
            jpeg = self.encode_frame(frame)
            if jpeg:
                with self.frame_lock:
                    self.latest_jpeg = jpeg
                    self.latest_frame_id += 1
                    self.capture_count += 1
            sleep_for = self.frame_delay - (time.time() - started)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def raw_loop(self):
        last_sent_id = 0
        while self.running:
            with self.frame_lock:
                frame_id = self.latest_frame_id
                jpeg = self.latest_jpeg
            if jpeg is None or frame_id == last_sent_id:
                time.sleep(0.005)
                continue
            if self.send_frame_bytes(jpeg, RAW_STREAM_ENDPOINT, self.raw_session, timeout=0.8):
                last_sent_id = frame_id
                self.raw_sent += 1
            else:
                self.failed += 1
                time.sleep(0.02)

    def ai_loop(self):
        last_mode_check = 0
        while self.running:
            started = time.time()
            now = time.time()
            if now - last_mode_check > 0.5:
                self.get_mode()
                last_mode_check = now
            with self.frame_lock:
                frame_id = self.latest_frame_id
                jpeg = self.latest_jpeg
            if self.current_mode == "raw":
                self.latest_ai_frame_id = frame_id
            elif jpeg and frame_id != self.latest_ai_frame_id:
                endpoint = ENDPOINTS.get(self.current_mode, ENDPOINTS["detect"])
                if self.send_frame_bytes(jpeg, endpoint, self.ai_session, timeout=3.0):
                    self.latest_ai_frame_id = frame_id
                    self.ai_sent += 1
                else:
                    self.failed += 1
            sleep_for = self.ai_frame_delay - (time.time() - started)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def run(self):
        print("=" * 64)
        print("Pathira Raspberry Camera -> Android APK")
        print(f"Phone server: {self.base_url}")
        print(f"preview={self.fps} fps ai={self.ai_fps} fps size={self.width}x{self.height} q={self.jpeg_quality} color={self.color_mode}")
        print("Open app -> Vision -> Play -> Glasses Camera")
        print("=" * 64)
        workers = [
            threading.Thread(target=self.capture_loop, daemon=True),
            threading.Thread(target=self.raw_loop, daemon=True),
            threading.Thread(target=self.ai_loop, daemon=True),
        ]
        for worker in workers:
            worker.start()
        last_status = 0
        while self.running:
            now = time.time()
            if now - last_status > 2:
                print(
                    f"mode={self.current_mode:<8} captured={self.capture_count:<6} raw={self.raw_sent:<6} ai={self.ai_sent:<5} failed={self.failed:<4} ptt={'1' if self.ptt_pressed() else '0'}"
                )
                last_status = now
            time.sleep(0.2)

    def stop(self):
        self.running = False
        try:
            if self.picam2:
                self.picam2.stop()
        finally:
            self.control_session.close()
            self.raw_session.close()
            self.ai_session.close()


def main():
    parser = argparse.ArgumentParser(description="Pathira Raspberry camera sender for Android APK")
    parser.add_argument("--phone-host", "--server-host", dest="phone_host", default=DEFAULT_HOST)
    parser.add_argument("--phone-port", "--server-port", dest="phone_port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--ai-fps", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=55)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--color-mode", choices=["rgb", "bgr", "swap-rb"], default="rgb")
    args = parser.parse_args()
    phone_host = discover_phone(args.phone_host, args.phone_port)
    client = PathiraRaspberryClient(
        phone_host,
        args.phone_port,
        args.fps,
        args.ai_fps,
        args.jpeg_quality,
        args.width,
        args.height,
        args.color_mode,
    )
    try:
        client.start_camera()
        client.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.stop()


if __name__ == "__main__":
    main()