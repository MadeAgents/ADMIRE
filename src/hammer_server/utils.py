# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Union
from PIL import Image, ImageDraw, ImageFont
from absl import logging
from android_world.agents.m3a_utils import validate_ui_element
from android_world.env import env_launcher, json_action
from android_world.env.interface import AsyncEnv
from android_world.env.representation_utils import UIElement
from android_world.registry import TaskRegistry

from dataclasses import asdict
from threading import Lock, Thread

import base64
import dataclasses
import gradio as gr
import io
import json
import numpy as np
import os
import portpicker
import re
import subprocess
import time
import uuid

from hammer_server.schema import IMAGE_URL, TEXT, TYPE, URL

AVD_NAME_PREFIX = "AndroidWorldAvd_"
DEVICE_TIMEOUT = 30

task_registry = TaskRegistry()
aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)


@dataclasses.dataclass
class DeviceInfo:
    avd_name: str
    pid: int = None
    device_name: str = None
    emulator_console_port: int = None
    grpc_port: int = None
    occupied: bool = False
    is_alive: bool = False
    emulator_setup_exception: str = None


class DeviceManager:
    def __init__(
        self, max_devices: int = 4, adb_path: str = None, crashed_device_restart: bool = True
    ):
        self.max_devices = max_devices
        self.devices: Dict[str, DeviceInfo] = {}
        self.adb_path = adb_path or "adb"
        self._lock = Lock()
        self.running = False
        self._monitor_thread = None
        self._crashed_device_restart = crashed_device_restart

    def start(self, num_devices_at_start: int = 0):
        self.running = True
        self.create_devices(num_devices=num_devices_at_start)
        # self._monitor_thread = Thread(target=self._monitor_devices, daemon=True)
        # self._monitor_thread.start()
        logging.info(f"# of devices: {len(self.devices)}")

    def _monitor_devices(self):
        """monitor devices"""
        logging.info("device monitor thread started")
        while self.running:
            time.sleep(5)
            to_restart = []
            alive_devices = get_devices()
            for device in self.devices.values():
                if device.device_name in alive_devices:
                    continue
                to_restart.append(device)
                self.devices[device.device_name].is_alive = False

            if len(to_restart) > 0:
                logging.warning(
                    f"device monitor thread - failed devices: {[d.device_name for d in to_restart]}"
                )
            if not self._crashed_device_restart:
                continue
            for device in to_restart:
                with self._lock:
                    avd_name = device.avd_name
                    device_name = device.device_name
                    emulator_console_port = device.emulator_console_port
                    grpc_port = device.grpc_port
                    device_info = self._create_device(
                        avd_name=avd_name,
                        emulator_console_port=emulator_console_port,
                        grpc_port=grpc_port,
                        emulator_setup=False,
                    )
                    self.devices[device_name].pid = device_info["pid"]
                    self.devices[device_name].occupied = False
                    self.devices[device_name].is_alive = True

        logging.info("device monitor thread exiting")

    def _get_occupied_devices(self):
        devices = []
        for device in self.devices:
            if not self.devices[device].occupied:
                continue
            if not self.devices[device].is_alive:
                continue
            devices.append(device)
        return devices

    def _get_available_devices(self):
        devices = []
        for device in self.devices:
            if self.devices[device].occupied:
                continue
            if not self.devices[device].is_alive:
                continue
            devices.append(device)
        return devices

    def _get_available_avds(self):
        avds = get_avds()
        occupied_avds = [d.avd_name for d in self.devices.values()]
        return list(set(avds) - set(occupied_avds))

    def _create_device(
        self, avd_name, emulator_console_port=None, grpc_port=None, emulator_setup=False
    ):
        emulator_console_port = emulator_console_port or _pick_port(
            5554, is_emulator_console_port=True
        )
        grpc_port = grpc_port or _pick_port(8554)
        options = [
            "-no-boot-anim",
            "-no-window",
            "-no-audio",
            "-no-snapshot",
            f"-port {emulator_console_port}",
            f"-grpc {grpc_port}",
            "-cores 4",
            "-memory 4096",
            # "-qemu -smp 2,threads=2",
            "-gpu off",
        ]
        if check_kvm_support():
            options.append("-accel on")
        else:
            options.append("-accel off")
        options = " ".join(options)
        logdir = "log/emulator"
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(
            logdir, f"{avd_name}-port_{emulator_console_port}-grpc_{grpc_port}.log"
        )
        process = subprocess.Popen(
            ["/bin/bash", "-c", f"emulator -avd {avd_name} {options}"],
            stdout=open(logfile, "a", buffering=1),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        logging.info(f"Creating emulator in subprocess PID: {process.pid}")

        device_name = f"emulator-{emulator_console_port}"
        attempts = 5
        while device_name not in get_devices():
            time.sleep(DEVICE_TIMEOUT)
            attempts -= 1
            if attempts == 0:
                raise RuntimeError("Creating emulator failed.")

        emulator_setup_exception = None
        if emulator_setup:
            try:
                env = env_launcher.load_and_setup_env(
                    console_port=emulator_console_port,
                    emulator_setup=True,
                    adb_path=self.adb_path,
                    grpc_port=grpc_port,
                    a11y_method="none",
                    install_a11y_forwarding_app=False,
                )
                env.close()
            except Exception as e:
                emulator_setup_exception = f"emulator setup error: {e}"
                logging.error(emulator_setup_exception)
        return {
            "avd_name": avd_name,
            "pid": process.pid,
            "device_name": device_name,
            "emulator_console_port": emulator_console_port,
            "grpc_port": grpc_port,
            "emulator_setup_exception": emulator_setup_exception,
        }

    def _create_avd(self):
        avd_name = f"{AVD_NAME_PREFIX}{uuid.uuid4().hex}"
        cmd = f"""avdmanager create avd -n {avd_name} -d pixel_6 -k "system-images;android-33;google_apis;x86_64" """
        cmd = cmd.strip()
        process = subprocess.Popen(
            ["/bin/bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.info(f"Creating avd in subprocess PID: {process.pid}")
        return_code = process.wait()
        logging.info(f"Subprocess return code: {return_code}")
        assert return_code == 0, "Creating avd failed."
        return avd_name

    def create_devices(self, num_devices: int = 0):
        if not num_devices:
            return
        assert num_devices + len(self.devices) <= self.max_devices
        available_avds = self._get_available_avds()
        num_available_avds = len(available_avds)
        if num_devices > num_available_avds:
            logging.warning(
                f"# of available avds: {num_available_avds}, but required {num_devices} avds"
            )
        num_created_devices = 0
        for avd_name in available_avds:
            if num_created_devices >= num_devices:
                break
            try:
                device_info = self._create_device(avd_name, emulator_setup=False)
            except Exception as e:
                logging.error(e)
                delete_avds(avd_name=avd_name)
                continue
            device = device_info["device_name"]
            self.devices[device] = DeviceInfo(**device_info, occupied=False, is_alive=True)
            num_created_devices += 1

        if num_created_devices < num_devices:
            logging.warning(
                f"# of created devices: {num_available_avds}, but required {num_devices} devices. new avds will be created."
            )

        while num_created_devices < num_devices:
            device_info = None
            avd_name = None
            try:
                avd_name = self._create_avd()
                device_info = self._create_device(avd_name, emulator_setup=True)
            except Exception as e:
                logging.error(e)
                if avd_name is not None:
                    delete_avds(avd_name=avd_name)
                break
            assert device_info is not None
            device = device_info["device_name"]
            self.devices[device] = DeviceInfo(**device_info, occupied=False, is_alive=True)
            num_created_devices += 1
        available_devices = self.get_available_devices()
        logging.info(
            f"# of devices: {len(self.devices)}, # of available devices: {len(available_devices)}"
        )

    def get_available_devices(self):
        with self._lock:
            devices = self._get_available_devices()
        return devices

    def get_occupied_devices(self):
        with self._lock:
            devices = self._get_occupied_devices()
        return devices

    def request_device(self) -> DeviceInfo:
        with self._lock:
            num_occupied_devices = len(self._get_occupied_devices())
            if num_occupied_devices >= self.max_devices:
                raise RuntimeError(
                    f"The maximum number of devices ({self.max_devices}) has been reached."
                )
            available_devices = self._get_available_devices()
            if len(available_devices) > 0:
                device = available_devices[0]
                self.devices[device].occupied = True
            else:
                while True:
                    available_avds = self._get_available_avds()
                    if available_avds:
                        avd_name = available_avds[0]
                        try:
                            device_info = self._create_device(avd_name, emulator_setup=False)
                            device = device_info["device_name"]
                            self.devices[device] = DeviceInfo(
                                **device_info, occupied=True, is_alive=True
                            )
                        except Exception as e:
                            logging.error(e)
                            # avd maybe damaged
                            delete_avds(avd_name=avd_name)
                            continue
                    else:
                        avd_name = self._create_avd()
                        device_info = self._create_device(avd_name, emulator_setup=True)
                        device = device_info["device_name"]
                        self.devices[device] = DeviceInfo(
                            **device_info, occupied=True, is_alive=True
                        )
                    break
        available_devices = self.get_available_devices()
        logging.info(
            f"# of devices: {len(self.devices)}, # of available devices: {len(available_devices)}"
        )
        return self.devices[device]

    def release_device(self, device):
        with self._lock:
            self.devices[device].occupied = False
        available_devices = self.get_available_devices()
        logging.info(
            f"# of devices: {len(self.devices)}, # of available devices: {len(available_devices)}"
        )

    def release_all_devices(self):
        with self._lock:
            for device in self.devices:
                self.devices[device].occupied = False
        available_devices = self.get_available_devices()
        logging.info(
            f"# of devices: {len(self.devices)}, # of available devices: {len(available_devices)}"
        )

    def close(self):
        self.running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3 * DEVICE_TIMEOUT)
        for device_process_info in self.devices.values():
            device_name = device_process_info.pid
            kill_emulator(device_name=device_name, adb_path=self.adb_path)


def check_kvm_support():
    with open("/proc/cpuinfo", "r") as f:
        cpuinfo = f.read()
    #  vmx (Intel) or svm (AMD)
    if not ("vmx" in cpuinfo or "svm" in cpuinfo):
        return False
    if not os.path.exists("/dev/kvm"):
        return False
    return True


def _pick_port(port, is_emulator_console_port: bool = False) -> int:
    if portpicker.is_port_free(port):
        return port
    else:
        if is_emulator_console_port:
            for p in range(5554, 5586, 2):
                if portpicker.is_port_free(p):
                    return p
        return portpicker.pick_unused_port()


def build_emulator():
    avd_name = f"{AVD_NAME_PREFIX}{uuid.uuid4().hex}"
    cmd = f"""avdmanager create avd -n {avd_name} -d pixel_6 -k "system-images;android-33;google_apis;x86_64" """
    cmd = cmd.strip()
    process = subprocess.Popen(
        ["/bin/bash", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logging.info(f"Creating avd in subprocess PID: {process.pid}")
    return_code = process.wait()
    logging.info(f"Subprocess return code: {return_code}")
    assert return_code == 0, "Creating avd failed."

    emulator_console_port = _pick_port(5554)
    grpc_port = _pick_port(8554)
    process = subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            f"emulator -avd {avd_name} -no-window -no-audio -no-snapshot -port {emulator_console_port} -grpc {grpc_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logging.info(f"Creating emulator in subprocess PID: {process.pid}")

    device_name = f"emulator-{emulator_console_port}"
    attempts = 5
    while device_name not in get_devices():
        time.sleep(15)
        attempts -= 1
        if attempts == 0:
            raise RuntimeError("Creating emulator failed.")

    return {
        "avd_name": avd_name,
        "pid": process.pid,
        "device_name": device_name,
        "emulator_console_port": emulator_console_port,
        "grpc_port": grpc_port,
    }


def create_avds():
    avd_name = f"{AVD_NAME_PREFIX}{uuid.uuid4().hex}"
    cmd = f"""avdmanager create avd -n {avd_name} -d pixel_6 -k "system-images;android-33;google_apis;x86_64" """
    cmd = cmd.strip()
    process = subprocess.Popen(
        ["/bin/bash", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logging.info(f"Creating avd in subprocess PID: {process.pid}")
    return_code = process.wait()
    logging.info(f"Subprocess return code: {return_code}")
    assert return_code == 0, "Creating avd failed."
    return avd_name


def get_devices():
    devices = []
    try:
        result = subprocess.run(
            ["/bin/bash", "-c", "adb devices"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e.stderr}")
        return devices
    for device in result.stdout.strip().split("\n")[1:]:
        parts = device.strip().split("\t")
        if len(parts) == 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def get_avds():
    try:
        result = subprocess.run(
            ["/bin/bash", "-c", "avdmanager list avd"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e.stderr}")
    avds = re.findall(
        pattern=f"Name: ({AVD_NAME_PREFIX}.*?)(?=Device:)",
        string=result.stdout,
        flags=re.DOTALL,
    )
    return [avd.strip() for avd in avds]


def delete_avds(avd_name=None):
    if avd_name:
        avds = [avd_name]
    else:
        try:
            result = subprocess.run(
                ["/bin/bash", "-c", "avdmanager list avd"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Error: {e.stderr}")
        avds = re.findall(
            pattern=f"Name: ({AVD_NAME_PREFIX}.*?)(?=Device:)",
            string=result.stdout,
            flags=re.DOTALL,
        )
    for avd in avds:
        avd_name = avd.strip()
        try:
            result = subprocess.run(
                ["/bin/bash", "-c", f"avdmanager delete avd -n {avd_name}"],
                check=True,
                capture_output=True,
                text=True,
            )
            logging.info(f"{result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error: {e.stderr}")


def force_kill_emulator(pid):
    try:
        result = subprocess.run(
            ["/bin/bash", "-c", f"kill -9 {pid}"],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.info(f"{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e.stderr}")


def kill_emulator(device_name, adb_path="adb"):
    try:
        result = subprocess.run(
            ["/bin/bash", "-c", f"{adb_path} -s {device_name} emu kill"],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.info(f"{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e.stderr}")


def get_task_list():
    return list(aw_registry.keys())


def get_task(task, env: AsyncEnv, params=None):
    task_cls = aw_registry.get(task)
    if params is None:
        params = task_cls.generate_random_params()
    logging.info(f"task params: {params}")
    task = task_cls(params)
    task.initialize_task(env)
    return task


def get_action_types():
    return list(json_action._ACTION_TYPES)


def get_action_param_prompt(action_type: str):
    prompt = ""
    if action_type in [
        json_action.CLICK,
        json_action.DOUBLE_TAP,
    ]:
        prompt = """请使用 JSON 格式输入屏幕坐标（整型）。
```json
{
    "index": 元素索引，整型
}
```
"""
    elif action_type == json_action.SWIPE:
        prompt = """请使用 JSON 格式。
```json
{
    "index": 元素索引，整型,
    "direction": "滑动方向，字符串类型，取值范围："up", "down", "left", "right"
}
```
"""
    elif action_type == json_action.INPUT_TEXT:
        prompt = """请使用 JSON 格式。
```json
{
    "index": 元素索引，整型,
    "text": "输入文本，字符串类型"
}
```
"""
    elif action_type == json_action.SCROLL:
        prompt = """请使用 JSON 格式。
```json
{
    "direction": "滑动方向，字符串类型，取值范围："up", "down", "left", "right"
}
```
"""
    elif action_type == json_action.OPEN_APP:
        prompt = """请使用 JSON 格式。
```json
{
    "app_name": "APP 名称，字符串类型"
}
```
"""
    else:
        prompt = """请使用 JSON 格式。
```json
{}
```
"""
    return prompt


def image_to_base64(image: Union[np.ndarray, Image.Image]):
    assert isinstance(image, (np.ndarray, Image.Image))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue())
    # in py3, b64encode() returns bytes
    return f"""data:image/png;base64,{image_base64.decode("utf-8")}"""


def device_state_to_content(ui_elements, screenshot_base64):
    ui_elements = [json.dumps(asdict(elem)) for elem in ui_elements]
    content = [
        {TYPE: TEXT, TEXT: f"""[{",".join(ui_elements)}]"""},
        {
            TYPE: IMAGE_URL,
            IMAGE_URL: {URL: screenshot_base64},
        },
    ]
    return content


def base64_to_image(image_base64):
    # Remove data URI prefix if present
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]
    # Decode the Base64 string
    image_base64 = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_base64))
    return image


def screenshot_to_som_base64(
    screenshot: np.ndarray,
    ui_elements: list[UIElement],
) -> str:
    screenshot = Image.fromarray(screenshot)
    screen_size = screenshot.size

    draw = ImageDraw.Draw(im=screenshot)
    font = ImageFont.truetype("DejaVuSans.ttf", 100)
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, screen_size):
            position = ui_element.bbox_pixels.center
            draw.text(xy=position, text=str(index), fill="red", font=font)
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue())
    return f"""data:image/png;base64,{image_base64.decode("utf-8")}"""


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    return ip


def get_exception_message(e: Exception):
    return "".join([str(m) for m in e.args])
