"""Xbox controller reader via evdev uinput virtual device.

Data flow: Windows(pygame) -> TCP -> wsl_recv.py(uinput) -> evdev -> this module
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DEVICE_NAME = "Xbox Controller"


def find_xbox_device() -> Optional[str]:
    """Find the uinput virtual Xbox controller device."""
    import glob
    for dev_path in sorted(glob.glob("/dev/input/event*")):
        try:
            from evdev import InputDevice
            dev = InputDevice(dev_path)
            if "Xbox" in dev.name or "Microsoft" in dev.name:
                return dev_path
        except Exception:
            continue
    try:
        data = open("/proc/bus/input/devices").read()
        for line in data.split("\n"):
            if line.startswith("H: Handlers=") and "event" in line:
                for part in line.split():
                    if part.startswith("event"):
                        return f"/dev/input/{part}"
    except Exception:
        pass
    return None


@dataclass
class GamepadState:
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0
    btn_a: bool = False
    btn_b: bool = False
    btn_x: bool = False
    btn_y: bool = False
    btn_lb: bool = False
    btn_rb: bool = False
    btn_select: bool = False
    btn_start: bool = False
    btn_xbox: bool = False
    btn_ls: bool = False
    btn_rs: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_axis(self, code: int, value: int):
        with self._lock:
            if code == 0:
                self.left_x = value / 32767.0 if abs(value) > 8 else 0.0
            elif code == 1:
                self.left_y = -value / 32767.0 if abs(value) > 8 else 0.0
            elif code == 3:
                self.right_x = value / 32767.0 if abs(value) > 8 else 0.0
            elif code == 4:
                self.right_y = -value / 32767.0 if abs(value) > 8 else 0.0
            elif code == 2:
                self.lt = value / 1023.0
            elif code == 5:
                self.rt = value / 1023.0
            elif code == 16:
                self.hat_x = value
            elif code == 17:
                self.hat_y = -value

    def update_button(self, code: int, pressed: int):
        btn_map = {
            304: "btn_a", 305: "btn_b", 307: "btn_x", 308: "btn_y",
            310: "btn_lb", 311: "btn_rb", 314: "btn_select", 315: "btn_start",
            316: "btn_xbox", 317: "btn_ls", 318: "btn_rs",
        }
        name = btn_map.get(code)
        if name:
            with self._lock:
                setattr(self, name, bool(pressed))

    def snapshot(self) -> dict:
        with self._lock:
            return {k: getattr(self, k) for k in (
                "left_x", "left_y", "right_x", "right_y",
                "lt", "rt", "hat_x", "hat_y",
                "btn_a", "btn_b", "btn_x", "btn_y",
                "btn_lb", "btn_rb", "btn_select", "btn_start",
                "btn_ls", "btn_rs",
            )}


class GamepadReader:
    """Background thread reads evdev events, provides GamepadState."""

    def __init__(self, device_path: str = None):
        self.state = GamepadState()
        self._thread = None
        self._running = False
        self._device_path = device_path
        self._connected = False
        self._retry_interval = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        if self._running:
            return
        now = time.time()
        if now < self._retry_interval:
            return
        self._retry_interval = now + 2.0
        self._device_path = find_xbox_device()
        if self._device_path is None:
            print("[gamepad] 未找到 Xbox 手柄设备 (先运行 sudo python3 wsl_recv.py)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        time.sleep(0.3)
        if self._connected:
            print(f"[gamepad] ✓ 已连接: {self._device_path}")
        else:
            print(f"[gamepad] ⚠ 设备 {self._device_path} 连接中...")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def _read_loop(self):
        from evdev import InputDevice, ecodes
        try:
            dev = InputDevice(self._device_path)
            self._connected = True
            dev.grab()
            for event in dev.read_loop():
                if not self._running:
                    break
                if event.type == ecodes.EV_ABS:
                    self.state.update_axis(event.code, event.value)
                elif event.type == ecodes.EV_KEY:
                    self.state.update_button(event.code, event.value)
            dev.ungrab()
        except Exception as e:
            print(f"[gamepad] 读取错误: {e}")
        finally:
            self._connected = False
