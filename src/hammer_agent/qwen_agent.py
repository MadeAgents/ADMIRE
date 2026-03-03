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

"""Ref:
https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb
https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/utils/agent_function_call.py
"""

import math
from absl import logging
from openai import OpenAI
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt, Message, ContentItem
from qwen_agent.tools.base import BaseTool, register_tool
from typing import Tuple, Union

import json
import re

from hammer_server.client import HammerEnvClient
from hammer_server.utils import base64_to_image
from .utils.timer import timer

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

TEMPERATURE = 0


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `answer`: Output the answer.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "answer",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, `action=answer`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    @property
    def function(self) -> dict:  # Bad naming. It should be `function_info`.
        return {
            "name_for_human": self.name_for_human,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "args_format": self.args_format,
        }

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(coordinate=params["coordinate"])
        elif action == "long_press":
            return self._long_press(coordinate=params["coordinate"], time=params["time"])
        elif action == "swipe":
            return self._swipe(coordinate=params["coordinate"], coordinate2=params["coordinate2"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()

    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _system_button(self, button: str):
        raise NotImplementedError()

    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()


class Operator:
    def __init__(
        self,
        device_client: HammerEnvClient,
        max_steps=20,
        model_name: str = "Qwen2.5-VL-72B-Instruct",
        go_home_at_start: bool = False,
    ):
        self.device_client = device_client
        self.max_steps = max_steps
        self.model_name = model_name
        self.go_home_at_start = go_home_at_start

    def run(self, task):
        logging.info(f"task: {task}")

        device_error = False
        action_format_error = False
        is_successful = None
        timer_metrics = {}
        with timer(name="init_task", log=timer_metrics):
            obs = self.device_client.init_task(task)
        if self.go_home_at_start:
            _obs = self.device_client.step(
                action={"name": "navigate_home", "arguments": json.dumps({})}
            )
            obs.update(_obs)
        device_error = (not obs["is_alive"]) or (not obs["screenshot"])
        if device_error:
            return {
                "task": task,
                "goal": None,
                "trajectory": [],
                "is_successful": False,
                "timer": timer_metrics,
                "device_error": device_error,
                "action_format_error": action_format_error,
            }

        goal = obs["goal"]
        logging.info(f"goal: {goal}")
        screenshot_base64 = obs["screenshot"]
        exception = obs["exception"]
        screenshot = base64_to_image(screenshot_base64)
        screen_size = screenshot.size

        step = 0
        history = []
        actions = []
        while step < self.max_steps:
            messages = self._input_messages(
                task=goal,
                screenshot_base64=screenshot_base64,
                exception=exception,
                history=actions,
                screen_size=screen_size,
            )
            response = get_chat_completion(messages=messages, model_id=self.model_name)
            logging.info(f"response: {response}")

            history.append(
                {
                    "observation": screenshot_base64,
                    "exception": exception,
                    "response": response,
                    "action": None,
                }
            )
            try:
                action = _extract_action(response)
                actions.append(json.dumps(action, ensure_ascii=False))
                logging.debug(f"action: {action}")
                if action is not None:
                    action = _convert_action(action=action, screen_size=screen_size)
                history[-1]["action"] = json.dumps(action, ensure_ascii=False)
                # next step
                if action is None:
                    break
                if action["action_type"] == "status":
                    break
                action_type = action.pop("action_type")
                action_args = action
                action = {
                    "name": action_type,
                    "arguments": json.dumps(action_args, ensure_ascii=False),
                }
            except Exception as e:
                logging.error(f"action format error: {e}")
                action_format_error = True
                break

            try:
                with timer(name=f"step_{step}", log=timer_metrics):
                    obs = self.device_client.step(action=action)
                device_error = not obs["is_alive"]
                if device_error:
                    break
                screenshot_base64 = obs["screenshot"]
                exception = obs["exception"]
            except Exception as e:
                logging.error(f"device client error: {e}")
                break
            step += 1

        with timer(name=f"submit", log=timer_metrics):
            obs = self.device_client.submit()
        device_error = not obs["is_alive"]
        if not device_error:
            is_successful = obs["is_successful"]
        return {
            "task": task,
            "goal": goal,
            "trajectory": history,
            "is_successful": is_successful,
            "timer": timer_metrics,
            "device_error": device_error,
            "action_format_error": action_format_error,
        }

    def _input_messages(
        self,
        task,
        screenshot_base64,
        exception,
        history,
        screen_size,
        patch_size=14,
        merge_size=2,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    ):
        w, h = screen_size
        resized_height, resized_width = smart_resize(
            height=h,
            width=w,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        system_message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")])
            ],
            functions=[mobile_use.function],
            lang=None,
        )

        history = [f"Step {idx}: {action}" for idx, action in enumerate(history, start=1)]
        history = "; ".join(history)

        messages = []
        system_message = system_message[0].model_dump()
        logging.debug(f"system message: {system_message}\nhistory: {history}")
        messages.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            }
        )
        last_message_content = []
        if screenshot_base64:
            last_message_content.append(
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": screenshot_base64},
                }
            )
        if exception:
            last_message_content.append(
                {
                    "type": "text",
                    "text": f"Exception: {exception}",
                }
            )
        last_message_content.append(
            {
                "type": "text",
                "text": f"The user query: {task} (You have done the following operation on the current device): {history}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": last_message_content,
            }
        )
        return messages


class OperatorWithThinkingAndConclusion(Operator):
    def run(self, task):
        logging.info(f"task: {task}")

        device_error = False
        action_format_error = False
        is_successful = None
        timer_metrics = {}
        with timer(name="init_task", log=timer_metrics):
            obs = self.device_client.init_task(task)
        if self.go_home_at_start:
            _obs = self.device_client.step(
                action={"name": "navigate_home", "arguments": json.dumps({})}
            )
            obs.update(_obs)
        device_error = (not obs["is_alive"]) or (not obs["screenshot"])
        if device_error:
            return {
                "task": task,
                "goal": None,
                "trajectory": [],
                "is_successful": False,
                "timer": timer_metrics,
                "device_error": device_error,
                "action_format_error": action_format_error,
            }

        goal = obs["goal"]
        logging.info(f"goal: {goal}")
        screenshot_base64 = obs["screenshot"]
        exception = obs["exception"]
        screenshot = base64_to_image(screenshot_base64)
        screen_size = screenshot.size

        step = 0
        history = []
        conclusions = []
        while step < self.max_steps:
            messages = self._input_messages(
                task=goal,
                screenshot_base64=screenshot_base64,
                exception=exception,
                history=conclusions,
                screen_size=screen_size,
            )
            response = get_chat_completion(messages=messages, model_id=self.model_name)
            logging.info(f"response: {response}")

            history.append(
                {
                    "observation": screenshot_base64,
                    "exception": exception,
                    "response": response,
                    "action": None,
                }
            )
            try:
                action = _extract_action(response)
                conclusion = _extract_conclusion(response)
                conclusions.append(conclusion)
                logging.debug(f"action: {action}\nconclusion: {conclusion}")
                if action is not None:
                    action = _convert_action(action=action, screen_size=screen_size)
                history[-1]["action"] = json.dumps(action, ensure_ascii=False)
                # next step
                if action is None:
                    break
                if action["action_type"] == "status":
                    break
                action_type = action.pop("action_type")
                action_args = action
                action = {
                    "name": action_type,
                    "arguments": json.dumps(action_args, ensure_ascii=False),
                }
            except Exception as e:
                logging.error(f"action format error: {e}")
                action_format_error = True
                break

            try:
                with timer(name=f"step_{step}", log=timer_metrics):
                    obs = self.device_client.step(action=action)
                device_error = not obs["is_alive"]
                if device_error:
                    break
                screenshot_base64 = obs["screenshot"]
                exception = obs["exception"]
            except Exception as e:
                logging.error(f"device client error: {e}")
                break
            step += 1

        with timer(name=f"submit", log=timer_metrics):
            obs = self.device_client.submit()
        device_error = not obs["is_alive"]
        if not device_error:
            is_successful = obs["is_successful"]
        return {
            "task": task,
            "goal": goal,
            "trajectory": history,
            "is_successful": is_successful,
            "timer": timer_metrics,
            "device_error": device_error,
            "action_format_error": action_format_error,
        }

    def _input_messages(
        self,
        task,
        screenshot_base64,
        exception,
        history,
        screen_size,
        patch_size=14,
        merge_size=2,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    ):
        w, h = screen_size
        resized_height, resized_width = smart_resize(
            height=h,
            width=w,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        system_message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(
                    role="system",
                    content=[ContentItem(text="You are a helpful assistant.")],
                ),
            ],
            functions=[mobile_use.function],
            lang=None,
        )

        history = [f"Step {idx}: {action}" for idx, action in enumerate(history, start=1)]
        history = "; ".join(history)

        messages = []
        system_message = system_message[0].model_dump()
        logging.debug(f"system message: {system_message}\nhistory: {history}")
        messages.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            }
        )
        last_message_content = []
        if screenshot_base64:
            last_message_content.append(
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": screenshot_base64},
                }
            )
        if exception:
            last_message_content.append(
                {
                    "type": "text",
                    "text": f"Exception: {exception}",
                }
            )
        last_message_content.append(
            {
                "type": "text",
                "text": f"The user query: {task}\nTask progress (You have done the following operation on the current device): {history}\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": last_message_content,
            }
        )
        return messages


def _extract_action(response):
    action = None
    try:
        response = response.replace("📐", "</tool_call>")
        response = response.replace("⚗", "</tool_call>")
        matched = re.search(r"<tool_call>\n(.*?)\n</tool_call>", response, flags=re.DOTALL)
        if matched:
            action = matched.group(1)
            action = action.replace("<tool_call>", "").replace("</tool_call>", "")
            action = action.strip()
            action = json.loads(action) if action else None
    except Exception as e:
        logging.warning(f"parsing action field, {response}, {e}")
    return action


def _extract_conclusion(response):
    conclusion = None
    try:
        matched = re.search(r"<conclusion>\n(.*?)\n</conclusion>", response, flags=re.DOTALL)
        if matched:
            conclusion = matched.group(1)
            conclusion = conclusion.strip()
    except Exception as e:
        logging.warning(f"parsing conclusion field, {response}, {e}")
    return conclusion


def _convert_action(
    action,
    screen_size,
    patch_size=14,
    merge_size=2,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
):
    action = action["arguments"]
    action_type = action["action"]
    w, h = screen_size
    resized_height, resized_width = smart_resize(
        height=h,
        width=w,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    logging.debug(
        f"screen size: {screen_size}, resized screen size: {[resized_width, resized_height]}"
    )
    rescale_w = w / resized_width
    rescale_h = h / resized_height

    match action_type:
        case "key":
            keycode = action["text"]
            return {"action_type": "press_keyboard", "keycode": keycode}
        case "click":
            x, y = action["coordinate"]
            return {"action_type": "click", "x": int(x * rescale_w), "y": int(y * rescale_h)}
        case "long_press":
            x, y = action["coordinate"]
            time = action["time"]
            return {
                "action_type": "long_press",
                "x": int(x * rescale_w),
                "y": int(y * rescale_h),
                "duration": int(time) * 1000,
            }
        case "swipe":
            # to execute_adb_action "scroll" (src/hammer_world/env/actuation.py)
            # start_x, start_y = action["coordinate"]
            # end_x, end_y = action["coordinate2"]
            # dx = end_x - start_x
            # dy = end_y - start_y
            # if abs(dx) < abs(dy):
            #     if dy > 0:
            #         direction = "up"
            #     else:
            #         direction = "down"
            # else:
            #     if dx > 0:
            #         direction = "left"
            #     else:
            #         direction = "right"
            # return {"action_type": "scroll", "direction": direction}

            # to execute_adb_action "swipe" (src/hammer_world/env/actuation.py)
            touch_xy = action["coordinate"]
            touch_xy = [int(touch_xy[0] * rescale_w), int(touch_xy[1] * rescale_h)]
            lift_xy = action["coordinate2"]
            lift_xy = [int(lift_xy[0] * rescale_w), int(lift_xy[1] * rescale_h)]
            return {"action_type": "swipe", "touch_xy": touch_xy, "lift_xy": lift_xy}
        case "type":
            text = action["text"]
            return {"action_type": "input_text", "text": text}
        case "answer":
            text = action["text"]
            return {"action_type": "answer", "text": text}
        case "system_button":
            button = action["button"]
            match button:
                case "Back":
                    return {"action_type": "navigate_back"}
                case "Home":
                    return {"action_type": "navigate_home"}
                case "Menu":
                    return {"action_type": "press_keyboard", "text": "KEYCODE_MENU"}
                case "Enter":
                    return {"action_type": "keyboard_enter"}
                case _:
                    return {"action_type": "unknown"}
        case "open":
            app_name = action["text"]
            return {"action_type": "open_app", "app_name": app_name}
        case "wait":
            time = action["time"]
            return {"action_type": "wait", "duration": int(time)}
        case "terminate":
            status = action["status"]
            return {"action_type": "status", "goal_status": status}
        case _:
            return {"action_type": "unknown"}


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def get_chat_completion(messages, client=None, model_id="Qwen2.5-VL-72B-Instruct"):
    client = client or OpenAI()
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=512,
        top_p=1,
        stream=False,
    )

    response = completion.choices[0].message.content
    return response


def set_temperature(temperature: float):
    global TEMPERATURE
    TEMPERATURE = temperature
    logging.info(f"temperature = {TEMPERATURE}")
