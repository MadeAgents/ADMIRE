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

from absl import logging
from android_world.env import env_launcher, json_action
from android_world.env.android_world_controller import A11yMethod
from android_world.env.representation_utils import UIElement
from android_world.env.interface import State as DeviceState
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import argparse
import json
import gradio as gr
import os
import pickle
import re
import time
import uuid
import numpy as np
from PIL import Image

from hammer_server.schema import ASSISTANT, CONTENT, DEVICE, ROLE
from hammer_server.log import setup_logger
from hammer_server.utils import (
    DeviceInfo,
    DeviceManager,
    get_action_param_prompt,
    get_action_types,
    get_exception_message,
    get_task,
    get_task_list,
    image_to_base64,
    screenshot_to_som_base64,
)
from hammer_trainer.utils.dataset.vision_utils import base64_to_image

logging.set_verbosity("info")
logging.use_python_logging(quiet=True)
logdir = "log/server"
os.makedirs(logdir, exist_ok=True)
_ = setup_logger(
    level=logging.INFO,
    file_info=os.path.join(logdir, "info.log"),
    file_err=os.path.join(logdir, "error.log"),
)

os.environ["GRPC_VERBOSITY"] = "ERROR"  # Only show errors
os.environ["GRPC_TRACE"] = "none"  # Disable tracing

_ADB_PATH = "/usr/lib/Android/Sdk/platform-tools/adb"
MAX_DEVICES = 24
NUM_DEVICES = 4
TIMEOUT = 60 * 10
STATE_WAITING = 2
GET_UI_ELEMENTS = False

SERV_DIR = Path(__file__).resolve().parent
with open(SERV_DIR / "css/block.css", "r") as f:
    block_css = f.read()

device_manager: DeviceManager = None
a11y_method: A11yMethod = A11yMethod.A11Y_FORWARDER_APP
install_a11y_forwarding_app: bool = a11y_method == A11yMethod.A11Y_FORWARDER_APP
get_ui_elements = False if a11y_method == A11yMethod.NONE else GET_UI_ELEMENTS
wait_to_stabilize = get_ui_elements
state_waiting = STATE_WAITING


class State:
    def __init__(self):
        self.device: DeviceInfo = None
        self.env = None
        self._is_alive = False
        self.messages: List[Dict] = []
        self.task = None
        self.task_id = None

    def request_device(self):
        auxiliaries = {}
        observation = DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries={})
        try:
            self.device = device_manager.request_device()
            logging.info(f"request_device: {self.device}")
            emulator_console_port = self.device.emulator_console_port
            grpc_port = self.device.grpc_port
            self.env = env_launcher.load_and_setup_env(
                console_port=emulator_console_port,
                emulator_setup=False,
                adb_path=_ADB_PATH,
                grpc_port=grpc_port,
                a11y_method=a11y_method,
                install_a11y_forwarding_app=install_a11y_forwarding_app,
            )
            observation = self.env.get_state(
                wait_to_stabilize=wait_to_stabilize, get_ui_elements=get_ui_elements
            )
            self._is_alive = self.device.is_alive
        except Exception as e:
            logging.error(f"request_device: {e}")
            self._is_alive = False
            auxiliaries["exception"] = get_exception_message(e)
        observation.auxiliaries.update(auxiliaries)
        return observation

    def init_task(self, task: str, params=None) -> DeviceState:
        auxiliaries = {}
        if not self.is_alive:
            auxiliaries["exception"] = "设备已经异常退出"
            return DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries=auxiliaries)

        observation = DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries={})
        try:
            self.task = get_task(task=task, env=self.env, params=params)
            self.task_id = uuid.uuid4().hex
            observation = self.env.get_state(
                wait_to_stabilize=wait_to_stabilize, get_ui_elements=get_ui_elements
            )
        except Exception as e:
            logging.error(f"task ({task}) init failed: {e}")
            auxiliaries["exception"] = get_exception_message(e)
        observation.auxiliaries.update(auxiliaries)
        self.messages.append({ROLE: DEVICE, CONTENT: observation})
        return observation

    def reset(self):
        if not self.is_alive:
            logging.warning(f"reset: session state is not alive, device info ({self.device})")
            if not self.device.is_alive:
                device_manager.release_device(device=self.device.device_name)
                self.device = device_manager.request_device()
                emulator_console_port = self.device.emulator_console_port
                grpc_port = self.device.grpc_port
                self.env = env_launcher.load_and_setup_env(
                    console_port=emulator_console_port,
                    emulator_setup=False,
                    adb_path=_ADB_PATH,
                    grpc_port=grpc_port,
                    a11y_method=a11y_method,
                    install_a11y_forwarding_app=install_a11y_forwarding_app,
                )
            elif self.device.is_alive and not self.device.occupied:
                emulator_console_port = self.device.emulator_console_port
                grpc_port = self.device.grpc_port
                self.env = env_launcher.load_and_setup_env(
                    console_port=emulator_console_port,
                    emulator_setup=False,
                    adb_path=_ADB_PATH,
                    grpc_port=grpc_port,
                    a11y_method=a11y_method,
                    install_a11y_forwarding_app=install_a11y_forwarding_app,
                )
                self.device.occupied = True
            else:
                pass
            self._is_alive = self.device.is_alive
        else:
            if self.task is not None:
                self.task.tear_down(self.env)

        logging.info(f"reset device: {self.device}")
        self.messages.clear()
        self.task = None
        self.task_id = None

    def release(self):
        if self.task is not None:
            self.task.tear_down(self.env)
        if self.env is not None:
            self.env.close()
        if self.device is not None:
            device = self.device.device_name
            device_manager.release_device(device=device)
        self.messages.clear()
        self._is_alive = False
        self.task = None
        self.task_id = None

    def get_current_screenshot_image(self):
        obs = self.env.get_state(wait_to_stabilize=False, get_ui_elements=False)
        screenshot = obs.pixels
        if isinstance(screenshot, np.ndarray):
            screenshot = Image.fromarray(screenshot)
        return screenshot

    def step(self, action) -> DeviceState:
        self.messages.append({ROLE: ASSISTANT, CONTENT: dict(action=action)})
        auxiliaries = {}
        if not self.is_alive:
            auxiliaries["exception"] = "设备已经异常退出"
            return DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries=auxiliaries)
        observation = DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries={})
        try:
            screenshot_bef = self.get_current_screenshot_image()
            self.env.execute_action(action, get_ui_elements=get_ui_elements)
            screenshot_aft = self.get_current_screenshot_image()
            if screenshot_bef == screenshot_aft:
                logging.info(f"state waiting for {state_waiting} seconds...")
                time.sleep(state_waiting)
            observation = self.env.get_state(
                wait_to_stabilize=wait_to_stabilize, get_ui_elements=get_ui_elements
            )
        except Exception as e:
            logging.error(f"step: {e}")
            auxiliaries["exception"] = get_exception_message(e)
        observation.auxiliaries.update(auxiliaries)
        self.messages.append({ROLE: DEVICE, CONTENT: observation})
        return observation

    def get_state(self) -> DeviceState:
        auxiliaries = {}
        if not self.is_alive:
            auxiliaries["exception"] = "设备已经异常退出"
            return DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries=auxiliaries)
        observation = DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries={})
        try:
            observation = self.env.get_state(
                wait_to_stabilize=wait_to_stabilize, get_ui_elements=get_ui_elements
            )
        except Exception as e:
            logging.error(f"get state: {e}")
            auxiliaries["exception"] = get_exception_message(e)
        observation.auxiliaries.update(auxiliaries)
        return observation

    def submit(self):
        auxiliaries = {}
        if not self.is_alive:
            auxiliaries["exception"] = "设备已经异常退出"
            auxiliaries["is_successful"] = False
            return DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries=auxiliaries)
        task = self.task
        env = self.env
        is_successful = 0
        observation = DeviceState(pixels=None, forest=None, ui_elements=[], auxiliaries={})
        try:
            is_successful = task.is_successful(env)
            observation = self.env.get_state(
                wait_to_stabilize=wait_to_stabilize, get_ui_elements=get_ui_elements
            )
        except Exception as e:
            auxiliaries["exception"] = get_exception_message(e)
        auxiliaries["is_successful"] = is_successful == 1
        observation.auxiliaries.update(auxiliaries)
        self.messages.append({ROLE: DEVICE, CONTENT: observation})
        return observation

    @property
    def is_alive(self):
        if self.device.is_alive == False:
            self._is_alive = False
        if self.device.occupied == False:
            self._is_alive = False
        return self._is_alive

    @property
    def device_screen_size(self):
        if not self.is_alive:
            return None
        return self.env.device_screen_size

    @property
    def device_name(self):
        if not self.is_alive:
            return None
        return self.device.device_name

    # def close(self):
    #     kill_emulator(self.emulator_process.get("pid"))
    #     delete_avds(self.emulator_process.get("avd_name"))


def _parse_action(action_type, action_args, ui_elements: List[UIElement]):
    """Ref: android_world/android_world/agents/random_agent.py
    function _generate_random_action

    support action space: qwen25vl
    """
    exception = ""
    # action arguments
    pattern = r"```json\s*({.*?})\s*```"
    match = re.search(pattern, action_args, re.DOTALL)
    if match:
        action_args = match.group(1)
    try:
        action_args = json.loads(action_args)
    except json.JSONDecodeError as e:
        exception += (
            f"json decoding error: {e}. action_type:{action_type}, action_args: {action_args}"
        )
        logging.error(exception)
        return {"action": None, "exception": exception}

    try:
        action = {"action_type": action_type}
        # if action_type in [
        #     json_action.CLICK,
        #     json_action.DOUBLE_TAP,
        #     json_action.SWIPE,
        #     json_action.INPUT_TEXT,
        # ]:
        #     if index := action_args.get("index"):
        #         ui_elem = ui_elements[index]
        #         xy = ui_elem.bbox_pixels.center
        #     else:
        #         xy = (action_args["x"], action_args["y"])
        #     action["x"], action["y"] = xy
        #     if action_type == json_action.INPUT_TEXT:
        #         action["text"] = action_args["text"]
        #     elif action_type == json_action.SWIPE:
        #         action["direction"] = action_args["direction"]
        # elif action_type == json_action.SCROLL:
        #     action["direction"] = action_args["direction"]
        # else:
        #     action.update(action_args)
        action.update(action_args)
    except Exception as e:
        exception += f"action error: {e}. action_type:{action_type}, action_args: {action_args}"
        logging.error(exception)
        return {"action": None, "exception": exception}

    return {"action": json_action.JSONAction(**action), "exception": exception}


def _get_device_info_html(device_info, screenshot_base64, ui_elements=[]):
    layout = ""
    if ui_elements:
        ui_elements = [asdict(element) for element in ui_elements]
        layout = f"""<details>
            <summary>UI 布局：{json.dumps(ui_elements, default=str)}</summary>
        </details>"""
    device_info_html = f"""<div>
        <p>设备可用：{"是" if device_info["is_alive"] else "否"}</p>
        <p>设备名称：{device_info["device_name"]}</p>
        <p>屏幕尺寸：{device_info["screen_size"]}</p>
        <p><img src="{screenshot_base64}" alt="当前屏幕" style="max-width: 100%; height: auto;"/></p>
        <p>异常消息：{device_info.get("exception")}</p>
        {layout}
    </div>"""

    return device_info_html


def update_action_param_prompt(action_type):
    prompt = get_action_param_prompt(action_type=action_type)
    return gr.update(value=prompt, submit_btn=True, interactive=True)


def message_pair_to_chatbot(message_pair):
    return (
        message_pair[0],
        f"""<div class="image-container"><img src="{message_pair[1]}" /></div>""",
    )


def request_device(_state: State, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"request_device. ip: {ip}")
    _state = State()
    observation = _state.request_device()
    device_info = {}
    device_info["device_name"] = _state.device_name
    device_info["screen_size"] = _state.device_screen_size
    device_info["is_alive"] = _state.is_alive
    device_info["exception"] = observation.auxiliaries.get("exception", "")

    if not _state.is_alive:
        screenshot_base64 = ""
        outputs = [_state]
        # request device button
        outputs.append(gr.update())
        # task selector
        outputs.append(gr.update(interactive=False))
        # device info html
        outputs.append(
            gr.update(
                value=_get_device_info_html(
                    device_info=device_info, screenshot_base64=screenshot_base64
                )
            )
        )
        # release button
        outputs.append(gr.update(interactive=True))
        return outputs

    # observation
    screenshot = observation.pixels
    ui_elements = observation.ui_elements
    screenshot_base64 = image_to_base64(screenshot)

    outputs = [_state]
    # request device button
    outputs.append(gr.update(interactive=False))
    # task selector
    outputs.append(gr.update(interactive=True))
    # device info html
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info,
                screenshot_base64=screenshot_base64,
                ui_elements=ui_elements,
            )
        )
    )
    # release button
    outputs.append(gr.update(interactive=True))
    return outputs


def init_task(
    task: str, task_param_file: str, _state: State, _chatbot: gr.Chatbot, request: gr.Request
):
    ip = get_ip(request)
    logging.info(f"init_task. ip: {ip}, task: {task}, task_param_file: {task_param_file}")

    if task is None:
        return [gr.update() for _ in range(9)]

    task_params = None
    if task_param_file is not None and os.path.isfile(task_param_file):
        with open(task_param_file, mode="rb") as f:
            task_params = pickle.load(f)

    _state.reset()
    _chatbot.clear()

    observation = _state.init_task(task=task, params=task_params)
    device_info = {}
    device_info["device_name"] = _state.device_name
    device_info["screen_size"] = _state.device_screen_size
    device_info["is_alive"] = _state.is_alive and (_state.task is not None)
    device_info["exception"] = observation.auxiliaries.get("exception", "")

    screenshot_base64 = ""
    screenshot_som_base64 = ""
    if (not _state.is_alive) or (_state.task is None):
        outputs = [_state]
        # chatbot
        message_pair = ("", screenshot_som_base64)
        _chatbot.append(message_pair_to_chatbot(message_pair=message_pair))
        outputs.append(_chatbot)
        # task textbox
        task = None if _state.task is None else _state.task.goal
        outputs.append(gr.update(value=task, show_copy_button=True))
        # action type radio
        outputs.append(gr.update(value=None, interactive=False))
        # action param textbox
        outputs.append(gr.update(value=None, interactive=False))
        # device info html
        outputs.append(
            gr.update(
                value=_get_device_info_html(
                    device_info=device_info, screenshot_base64=screenshot_base64
                )
            )
        )
        # reset button
        outputs.append(gr.update(interactive=True))
        # submit button
        outputs.append(gr.update(interactive=False))
        # task result textbox
        outputs.append(gr.update(value=None))
        return outputs

    # observation
    screenshot = observation.pixels
    ui_elements = observation.ui_elements
    if screenshot is not None:
        screenshot_base64 = image_to_base64(screenshot)
        screenshot_som_base64 = screenshot_to_som_base64(
            screenshot=screenshot, ui_elements=ui_elements
        )

    outputs = [_state]
    # chatbot
    message_pair = ("", screenshot_som_base64)
    _chatbot.append(message_pair_to_chatbot(message_pair=message_pair))
    outputs.append(_chatbot)
    # task textbox
    task = _state.task.goal
    outputs.append(gr.update(value=task, show_copy_button=True))
    # action type radio
    outputs.append(gr.update(value=None, interactive=True))
    # action param textbox
    outputs.append(gr.update(value=None))
    # device info html
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info,
                screenshot_base64=screenshot_base64,
                ui_elements=ui_elements,
            )
        )
    )
    # reset button
    outputs.append(gr.update(interactive=True))
    # submit button
    outputs.append(gr.update(interactive=True))
    # task result textbox
    outputs.append(gr.update(value=None))
    return outputs


def device_step(
    _action_type, _action_args, _state: State, _chatbot: gr.Chatbot, request: gr.Request
):
    ip = get_ip(request)
    logging.info(f"device_step. ip: {ip}, action_type: {_action_type}, action_args: {_action_args}")
    ui_elements = _state.messages[-1][CONTENT].ui_elements

    action = _parse_action(
        action_type=_action_type, action_args=_action_args, ui_elements=ui_elements
    )
    action, exception = action["action"], action["exception"]
    if action is None:
        device_info = {}
        device_info["device_name"] = ""
        device_info["screen_size"] = ""
        device_info["is_alive"] = True
        device_info["exception"] = exception
        message_pair = (f"action_type: {_action_type}, action_args: {_action_args}", "")
        _chatbot.append(message_pair_to_chatbot(message_pair))
        return (
            # state
            gr.update(),
            # chatbot
            _chatbot,
            # action type radio
            gr.update(value=None),
            # action param textbox
            gr.update(value=None),
            # device info html
            gr.update(value=_get_device_info_html(device_info=device_info, screenshot_base64="")),
            # submit button
            gr.update(),
            # task result textbox
            gr.update(value=None),
        )

    observation = _state.step(action=action)
    device_info = {}
    device_info["device_name"] = _state.device_name
    device_info["screen_size"] = _state.device_screen_size
    device_info["is_alive"] = _state.is_alive
    device_info["exception"] = observation.auxiliaries.get("exception", "")
    screenshot_base64 = ""
    screenshot_som_base64 = ""

    if not _state.is_alive:
        message_pair = (f"action: {action.json_str()}", screenshot_base64)
        _chatbot.append(message_pair_to_chatbot(message_pair))
        return (
            # state
            gr.update(),
            # chatbot
            _chatbot,
            # action type radio
            gr.update(value=None, interactive=False),
            # action param textbox
            gr.update(value=None, interactive=False),
            # device info html
            gr.update(
                value=_get_device_info_html(
                    device_info=device_info, screenshot_base64=screenshot_base64
                )
            ),
            # submit button
            gr.update(interactive=False),
            # task result textbox
            gr.update(value=None),
        )

    screenshot = observation.pixels
    ui_elements = observation.ui_elements
    if screenshot is not None:
        screenshot_base64 = image_to_base64(screenshot)
        screenshot_som_base64 = screenshot_to_som_base64(
            screenshot=screenshot, ui_elements=ui_elements
        )

    # state
    outputs = [_state]
    # chatbot
    message_pair = (f"action: {action.json_str()}", screenshot_som_base64)
    _chatbot.append(message_pair_to_chatbot(message_pair=message_pair))
    outputs.append(_chatbot)
    # action type radio
    outputs.append(gr.update(value=None))
    # action param textbox
    outputs.append(gr.update(value=None))
    # device info html
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info,
                screenshot_base64=screenshot_base64,
                ui_elements=ui_elements,
            )
        )
    )
    # submit button
    outputs.append(gr.update(interactive=True))
    # task result textbox
    outputs.append(gr.update(value=None))
    return outputs


def submit_task(_state: State, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"submit_task. ip: {ip}")

    observation = _state.submit()
    device_info = {}
    device_info["device_name"] = _state.device_name
    device_info["screen_size"] = _state.device_screen_size
    device_info["is_alive"] = _state.is_alive
    device_info["exception"] = observation.auxiliaries.get("exception", "")
    result = observation.auxiliaries.get("is_successful", False)

    if not _state.is_alive:
        screenshot_base64 = ""
        return (
            # device info html
            gr.update(
                value=_get_device_info_html(
                    device_info=device_info, screenshot_base64=screenshot_base64
                )
            ),
            # task result textbox
            gr.update(value="设备异常"),
        )
    # observation
    screenshot = observation.pixels
    ui_elements = observation.ui_elements
    screenshot_base64 = "" if screenshot is None else image_to_base64(screenshot)
    result = "任务成功 ✅" if result else "任务失败 ❌"
    return (
        # device info html
        gr.update(
            value=_get_device_info_html(
                device_info=device_info,
                screenshot_base64=screenshot_base64,
                ui_elements=ui_elements,
            )
        ),
        # task result textbox
        gr.update(value=result),
    )


def release_device(_state: State, _chatbot: gr.Chatbot, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"release_device. ip: {ip}")

    if _state is None:
        pass
    else:
        _state.release()
        _state = None
    _chatbot.clear()
    outputs = [_state]
    # chatbot
    outputs.append(_chatbot)
    # request device button
    outputs.append(gr.update(interactive=True))
    # task selector
    outputs.append(gr.update(value=None))
    # task textbox
    outputs.append(gr.update(value=None, show_copy_button=False))
    # action type radio
    outputs.append(gr.update(value=None, interactive=False))
    # action param textbox
    outputs.append(gr.update(value=None, interactive=False))
    # device info html
    outputs.append(gr.update(value=None))
    # reset button
    outputs.append(gr.update(interactive=False))
    # submit button
    outputs.append(gr.update(interactive=False))
    # task result textbox
    outputs.append(gr.update(value=None))
    # release button
    outputs.append(gr.update(interactive=False))
    return outputs


def get_occupied_devices(request: gr.Request):
    ip = get_ip(request)
    logging.info(f"get_occupied_devices. ip: {ip}")
    devices = device_manager.get_occupied_devices()
    devices = [d for d in devices]

    # occupied deices checkbox
    return gr.update(choices=devices, value=None, interactive=bool(devices))


def update_selected_occupied_devices(selected_occupied_devices):
    return gr.Button(interactive=bool(selected_occupied_devices))


def release_selected_occupied_devices(
    _state: State, selected_occupied_devices, request: gr.Request
):
    ip = get_ip(request)
    logging.info(f"release_selected_occupied_devices. ip: {ip}")
    logging.info(f"# of devices ({len(selected_occupied_devices)}) will be released.")

    device_update = None
    for device in selected_occupied_devices:
        if _state is not None and _state.device is not None and _state.device.device_name == device:
            _state.release()
            device_update = True
        else:
            device_manager.release_device(device)

    devices = device_manager.get_occupied_devices()
    devices = [d for d in devices]

    # state
    outputs = [_state]
    # request device button
    outputs.append(gr.update(interactive=True) if device_update else gr.update())
    # release button
    outputs.append(gr.update(interactive=False) if device_update else gr.update())
    # occupied deices checkbox
    outputs.append(gr.update(choices=devices, value=None, interactive=bool(devices)))
    # release selected devices button
    outputs.append(gr.update(interactive=True) if devices else gr.update(interactive=False))
    return outputs


def force_release_devices(request: gr.Request):
    ip = get_ip(request)
    logging.info(f"force_release_devices. ip: {ip}")
    device_manager.release_all_devices()

    outputs = []
    # request device button
    outputs.append(gr.update(interactive=True))
    # release button
    outputs.append(gr.update(interactive=False))
    # occupied deices checkbox
    outputs.append(gr.update(choices=[], value=None, interactive=False))
    # release selected devices button
    outputs.append(gr.update(interactive=False))
    return outputs


def _build_device_interface_tab():
    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                type="tuples",
                label="Android Device",
                height=900,
                show_copy_button=True,
                show_copy_all_button=True,
            )
            action_type_radio = gr.Radio(
                choices=get_action_types(), label="Next Action Type", interactive=False
            )
            action_param_textbox = gr.Textbox(
                lines=10, label="Next Action Parameters", submit_btn=False, interactive=False
            )
        with gr.Column(scale=1):
            request_device_btn = gr.Button("🧹 Request Device (申请设备)", interactive=True)
            task_list = get_task_list()
            task_selector = gr.Dropdown(
                choices=task_list, value=None, label="Select a task", interactive=False
            )
            task_param_file = gr.File(
                label="upload param file",
                file_count="single",
                file_types=[".pkl"],
                type="filepath",
                visible=False,
            )
            task_textbox = gr.Textbox(lines=2, label="Task", interactive=False)
            device_info_html = gr.HTML(
                value="<div>no device</div>",
                label="Device Info",
                container=True,
                show_label=True,
            )
            reset_btn = gr.Button("🧹 Reset (重置任务)", interactive=False)
            submit_btn = gr.Button("🚀 Submit (提交任务)", interactive=False)
            task_result_textbox = gr.Textbox(label="Task Result", interactive=False)
            release_btn = gr.Button("💨 Release (释放设备)", interactive=False)
    return (
        chatbot,
        action_type_radio,
        action_param_textbox,
        request_device_btn,
        task_selector,
        task_param_file,
        task_textbox,
        device_info_html,
        reset_btn,
        submit_btn,
        task_result_textbox,
        release_btn,
    )


def _build_device_manager_tab():
    with gr.Row():
        occupied_deices_checkbox = gr.CheckboxGroup(
            choices=None, label="已占用的设备", interactive=False
        )
    with gr.Row():
        get_occupied_deices_btn = gr.Button(
            "Get Occupied Deivces (获取已占用的设备)", interactive=True
        )
        release_selected_devices_btn = gr.Button(
            "Release Selected Deivces (释放选定的设备)", interactive=False
        )
    with gr.Row():
        force_release_btn = gr.Button("⚠️☠️ Force Release (强制释放设备)", interactive=True)
    return (
        occupied_deices_checkbox,
        get_occupied_deices_btn,
        release_selected_devices_btn,
        force_release_btn,
    )


def _build_demo():
    notice_markdown = f"""# 📱 Android 移动设备动态交互环境"""
    state = gr.State()

    with gr.Blocks(css=block_css) as demo:
        gr.Markdown(notice_markdown, elem_id="notice_markdown")
        with gr.Tab("移动设备交互"):
            (
                chatbot,
                action_type_radio,
                action_param_textbox,
                request_device_btn,
                task_selector,
                task_param_file,
                task_textbox,
                device_info_html,
                reset_btn,
                submit_btn,
                task_result_textbox,
                release_btn,
            ) = _build_device_interface_tab()
        with gr.Tab("移动设备管理"):
            (
                occupied_deices_checkbox,
                get_occupied_deices_btn,
                release_selected_devices_btn,
                force_release_btn,
            ) = _build_device_manager_tab()

        # device interaction
        state = gr.State()

        request_device_btn.click(
            fn=request_device,
            inputs=[state],
            outputs=[state, request_device_btn, task_selector, device_info_html, release_btn],
            show_progress=True,
        )

        task_selector.change(
            fn=init_task,
            inputs=[task_selector, task_param_file, state, chatbot],
            outputs=[
                state,
                chatbot,
                task_textbox,
                action_type_radio,
                action_param_textbox,
                device_info_html,
                reset_btn,
                submit_btn,
                task_result_textbox,
            ],
            show_progress=True,
        )
        action_type_radio.change(
            fn=update_action_param_prompt,
            inputs=[action_type_radio],
            outputs=[action_param_textbox],
            show_progress=True,
        )
        action_param_textbox.submit(
            fn=device_step,
            inputs=[action_type_radio, action_param_textbox, state, chatbot],
            outputs=[
                state,
                chatbot,
                action_type_radio,
                action_param_textbox,
                device_info_html,
                submit_btn,
                task_result_textbox,
            ],
            show_progress=True,
        )
        reset_btn.click(
            fn=init_task,
            inputs=[task_selector, task_param_file, state, chatbot],
            outputs=[
                state,
                chatbot,
                task_textbox,
                action_type_radio,
                action_param_textbox,
                device_info_html,
                reset_btn,
                submit_btn,
                task_result_textbox,
            ],
            show_progress=True,
        )
        submit_btn.click(
            fn=submit_task,
            inputs=[state],
            outputs=[device_info_html, task_result_textbox],
            show_progress=True,
        )
        release_btn.click(
            fn=release_device,
            inputs=[state, chatbot],
            outputs=[
                state,
                chatbot,
                request_device_btn,
                task_selector,
                task_textbox,
                action_type_radio,
                action_param_textbox,
                device_info_html,
                reset_btn,
                submit_btn,
                task_result_textbox,
                release_btn,
            ],
            show_progress=True,
        )

        # device manager
        get_occupied_deices_btn.click(
            fn=get_occupied_devices, outputs=[occupied_deices_checkbox], show_progress=True
        )
        occupied_deices_checkbox.change(
            fn=update_selected_occupied_devices,
            inputs=[occupied_deices_checkbox],
            outputs=[release_selected_devices_btn],
            show_progress=True,
        )
        release_selected_devices_btn.click(
            fn=release_selected_occupied_devices,
            inputs=[state, occupied_deices_checkbox],
            outputs=[
                state,
                request_device_btn,
                release_btn,
                occupied_deices_checkbox,
                release_selected_devices_btn,
            ],
            show_progress=True,
        )
        force_release_btn.click(
            fn=force_release_devices,
            outputs=[
                request_device_btn,
                release_btn,
                occupied_deices_checkbox,
                release_selected_devices_btn,
            ],
            show_progress=True,
        )
    return demo


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


def main(args):
    global device_manager
    global a11y_method
    global install_a11y_forwarding_app
    global get_ui_elements
    global state_waiting
    global wait_to_stabilize

    a11y_method = args.a11y_method
    install_a11y_forwarding_app = a11y_method == A11yMethod.A11Y_FORWARDER_APP
    get_ui_elements = False if a11y_method == A11yMethod.NONE else args.get_ui_elements
    args.get_ui_elements = get_ui_elements
    wait_to_stabilize = get_ui_elements
    state_waiting = args.state_waiting

    logging.info(f"config: {args}")
    device_manager = DeviceManager(
        max_devices=args.max_devices,
        adb_path=_ADB_PATH,
        crashed_device_restart=args.crashed_device_restart,
    )
    device_manager.start(num_devices_at_start=args.num_devices)
    demo = _build_demo()
    demo.queue(default_concurrency_limit=args.concurrency_limit).launch(
        share=False,
        show_error=True,
        server_name=args.server_name,
        server_port=args.server_port,
        app_kwargs={
            "timeout_keep_alive": TIMEOUT,
            # "ws_ping_interval": 10,
            # "ws_ping_timeout": 30,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Android 移动设备动态交互环境")

    parser.add_argument("--max-devices", type=int, default=MAX_DEVICES)
    parser.add_argument("--num-devices", type=int, default=NUM_DEVICES)
    parser.add_argument("--crashed-device-restart", action="store_true", default=False)
    parser.add_argument(
        "--a11y-method",
        type=A11yMethod,
        choices=[i for i in A11yMethod],
        default=A11yMethod.A11Y_FORWARDER_APP,
    )
    parser.add_argument("--concurrency-limit", type=int, default=MAX_DEVICES)
    parser.add_argument("--state-waiting", type=float, default=STATE_WAITING)
    parser.add_argument("--get-ui-elements", action="store_true", default=False)

    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--show-error", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
