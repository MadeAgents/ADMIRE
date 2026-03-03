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

from gradio_client import Client, handle_file

import logging
import re
import json
from typing import List  

from android_world.env.representation_utils import UIElement, BoundingBox
from hammer_server.utils import get_task_list

logger = logging.getLogger(__file__)
TIMEOUT = 60 * 10


class HammerEnvClient:
    def __init__(self, src: str, timeout: int = None):
        self.client = Client(src=src, httpx_kwargs={"timeout": timeout or TIMEOUT})
        self.task = None
        self.goal = None

    def request_device(self):
        result = self.client.predict(api_name="/request_device")
        device_info = ""
        exception = ""
        try:
            device_info = result[1]["value"]
        except Exception as e:
            exception = f"request device error@client: {e}"
            logger.error(exception)

        return {
            "is_alive": get_device_state(observation=device_info),
            "screenshot": get_screenshot(observation=device_info),
            "exception": get_device_exception(observation=device_info) + exception,
        }

    def init_task(self, task: str, task_param_file: str = None):
        self.task = task
        assert task in get_task_list()
        if task_param_file is not None:
            task_param_file = handle_file(task_param_file)

        goal = ""
        device_info = ""
        screenshot_som = ""
        exception = ""
        try:
            result = self.client.predict(
                task=self.task, task_param_file=task_param_file, _chatbot=[], api_name="/init_task"
            )
            goal = result[1]["value"]
            device_info = result[4]["value"]
            screenshot_som = result[0][-1][1]
        except Exception as e:
            exception = f"init task error@client: {e}"
            logger.error(exception)
        self.goal = goal

        return {
            "goal": self.goal,
            "is_alive": get_device_state(observation=device_info),
            "screenshot": get_screenshot(observation=device_info),
            "screenshot_som": get_screenshot(observation=screenshot_som),
            "exception": get_device_exception(observation=device_info) + exception,
            "ui_elements": get_ui_elements(observation=device_info)
        }

    def step(self, action):
        """Actions
        - If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
        - If you think the task is not feasible (including cases like you don't have enough information or can not perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
        - Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
        - Click/tap on an element on the screen. Use the coordinates to indicate which element you want to click: `{{"action_type": "click", "x": <target_x>, "y": <target_y>}}`.
        - Long press on an element on the screen, similar with the click action above,use the coordinates to indicate which element you want to long press: `{{"action_type": "long_press", "x": <target_x>, "y": <target_y>}}`.
        - Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the coordinates to indicate the target text field: `{{"action_type": "input_text", "text": <text_input>, "x": <target_x>, "y": <target_y>}}`
        - Press the Enter key: `{{"action_type": "keyboard_enter"}}`
        - Navigate to the home screen: `{{"action_type": "navigate_home"}}`
        - Navigate back: `{{"action_type": "navigate_back"}}`
        - Scroll the screen or a scrollable UI element in one of the four directions, use the same coordinates as above if you want to scroll a specific UI element, leave it empty when scrolling the whole screen: `{{"action_type": "scroll", "direction": <up, down, left, right>, "x": <optional_target_x>, "y": <optional_target_y>}}`
        - Open an app (nothing will happen if the app is not installed): `{{"action_type": "open_app", "app_name": <name>}}`
        - Wait for the screen to update: `{{"action_type": "wait"}}`
        """
        device_info = ""
        screenshot_som = ""
        exception = ""
        try:
            result = self.client.predict(
                _action_type=action["name"],
                _action_args=action["arguments"],
                _chatbot=[],
                api_name="/device_step",
            )
            device_info = result[3]["value"]
            screenshot_som = result[0][-1][1]
        except Exception as e:
            exception = f"action step error@client: {e}"
            logger.error(exception)
        return {
            "is_alive": get_device_state(observation=device_info),
            "screenshot": get_screenshot(observation=device_info),
            "screenshot_som": get_screenshot(observation=screenshot_som),
            "exception": get_device_exception(observation=device_info) + exception,
            "ui_elements": get_ui_elements(observation=device_info)
        }

    def submit(self):
        device_info = ""
        is_successful = ""
        exception = ""
        try:
            result = self.client.predict(api_name="/submit_task")
            device_info = result[0]["value"]
            is_successful = result[1]["value"]
        except Exception as e:
            exception = f"submit error@client: {e}"
            logger.error(exception)
        return {
            "is_alive": get_device_state(observation=device_info),
            "is_successful": is_successful,
            "exception": get_device_exception(observation=device_info) + exception,
        }

    def get_available_tasks(self):
        return get_task_list()

    def close(self):
        _ = self.client.predict(_chatbot=[], api_name="/release_device")
        return

    def release_all_devices(self):
        _ = self.client.predict(api_name="/force_release_devices")
        return


def get_screenshot(observation):
    match = re.findall(pattern=r"""<img src="(.*?)" .*?/>""", string=observation, flags=re.DOTALL)

    image_base64 = None
    if match:
        image_base64 = match[0]
    return image_base64


def get_device_state(observation):
    match = re.findall(pattern=r"""<p>设备可用：(.*?)</p>""", string=observation, flags=re.DOTALL)

    is_alive = False
    if match:
        is_alive = match[0] == "是"
    return is_alive


def get_device_exception(observation):
    match = re.findall(pattern=r"""<p>异常消息：(.*?)</p>""", string=observation, flags=re.DOTALL)
    exception = ""
    if match:
        exception = match[0]
    return exception

def get_ui_elements(observation: str) -> List[UIElement]:
    match = re.findall(pattern=r"""<summary>UI 布局：(.*?)</summary>""", string=observation, flags=re.DOTALL)
    elements = []
    if match:
        json_str = match[0]
        data_list = json.loads(json_str)  
        for data in data_list:
            if data.get('bbox'):  
                data['bbox'] = BoundingBox(**data['bbox'])  
            if data.get('bbox_pixels'):  
                data['bbox_pixels'] = BoundingBox(**data['bbox_pixels'])  
            elements.append(UIElement(**data))
    return elements