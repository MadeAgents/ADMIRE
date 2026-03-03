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

# https://github.com/bytedance/UI-TARS/blob/main/codes
import ast
import copy
import json
import re
from qwen_vl_utils.vision_process import (
    IMAGE_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    smart_resize,
)

SYSTEM_PROMPT = """
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {width}x{height}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `clear_text`: Delete all characters (text or default Name) in the input field.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "clear_text", "system_button", "open", "wait", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type` and `action=open`.", "type": "string"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>


# Output Format
Thought: ... (Your thinking process, explain your reasoning step-by-step )
Action: ... (Your action description, provide a brief description of the chosen action)
<answer>
[{{"name": <function-name>, "arguments": <args-json-object>}}]
</answer>
""".strip()


MOBILE_USE_HAMMER = """The user query: {goal}\nTask progress (You have done the following operation on the current device): {history}.  """

OUTPUT_FORMAT = """# Output Format
Thought: ... (Your thinking process, explain your reasoning step-by-step )
Action: ... (Your action description, provide a brief description of the chosen action)
<answer>
[{{"name": <function-name>, "arguments": <args-json-object>}}]
</answer>
"""

def parse_action(params, screen_size, resized_screen_size):
    action = params["action"]
    width, height = screen_size
    resized_width, resized_height = resized_screen_size

    if action == "click":
        x = int(params["coordinate"][0] / resized_width * width)
        y = int(params["coordinate"][1] / resized_height * height)
        return {"action_type": "click", "x": x, "y": y}
    elif action == "long_press":
        x = int(params["coordinate"][0] / resized_width * width)
        y = int(params["coordinate"][1] / resized_height * height)
        return {"action_type": "long_press", "x": x, "y": y}
    elif action == "swipe":
        x1 = int(params["coordinate"][0] / resized_width * width)
        y1 = int(params["coordinate"][1] / resized_height * height)
        x2 = int(params["coordinate2"][0] / resized_width * width)
        y2 = int(params["coordinate2"][1] / resized_height * height)
        # return {"action_type": "swipe", "touch_xy": [x1, y1], "lift_xy": [x2, y2]}
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < abs(dy):
            if dy > 0:
                direction = "up"
            else:
                direction = "down"
        else:
            if dx > 0:
                direction = "left"
            else:
                direction = "right"
        return {"action_type": "scroll", "direction": direction, "x": x1, "y": y1}
    elif action == "type":
        return {"action_type": "input_text", "text": params["text"]}
    elif action == "system_button":
        button = params["button"]
        if button == "Back":
            return {"action_type": "navigate_back"}
        elif button == "Home":
            return {"action_type": "navigate_home"}
        elif button == 'Enter':
            return {"action_type": "keyboard_enter"}
        elif button == 'Menu':
            return {"action_type": "navigate_menu"}
        else:
            return {"action_type": "unknown"}
    elif action == "open":
        return {"action_type": "open_app", "app_name": params['text']}
    elif action == "wait":
        return {"action_type": "wait"}
    elif action == "terminate":
        return {"action_type": "status", "goal_status": params['status']}
    elif action == "answer":
        return {"action_type": "answer", "text": params["text"]}
    elif action == "clear_text":
        return {"action_type": "clear_text"}
    else:
        return {"action_type": "unknown"}


def parse_action_to_structure_output(
    text,
    factor,
    origin_resized_height,
    origin_resized_width,
    model_type="qwen25vl",
    max_pixels=MAX_PIXELS,
    min_pixels=MIN_PIXELS,
):
    text = text.strip()

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    # parse response
    thought_pattern = r'Thought: (.*?)(?=Action:|$)'
    action_desc_pattern = r'Action: (.*?)(?=<answer>|$)'
    answer_pattern = r'<answer>(.*?)</answer>'

    match = re.search(thought_pattern, text, re.DOTALL)
    thought = match.group(1).strip() if match else ''

    match = re.search(action_desc_pattern, text, re.DOTALL)
    action_desc = match.group(1).strip() if match else ''

    match = re.search(answer_pattern, text, re.DOTALL)
    answer = match.group(1).strip() if match else ''

    match = re.search(r'"arguments":\s*({[^{}]*})', answer)
    answer = json.loads(match.group(1)) if match else {}

    # parse action
    action = parse_action(
        answer, 
        (origin_resized_width, origin_resized_height), 
        (smart_resize_width, smart_resize_height)
    )
    return [{
        "thought": thought,
        "action_description": action_desc,
        "answer": answer,
        "action": action 
    }]

def extract_info_from_action_text(text):
    text = text.strip()

    thought_pattern = r'Thought: (.*?)(?=Action:|$)'
    action_desc_pattern = r'Action: (.*?)(?=<answer>|$)'
    answer_pattern = r'<answer>(.*?)</answer>'

    match = re.search(thought_pattern, text, re.DOTALL)
    thought = match.group(1).strip() if match else ''

    match = re.search(action_desc_pattern, text, re.DOTALL)
    action_desc = match.group(1).strip() if match else ''

    match = re.search(answer_pattern, text, re.DOTALL)
    answer = match.group(1).strip() if match else ''

    match = re.search(r'"arguments":\s*({[^{}]*})', answer)
    answer = json.loads(match.group(1)) if match else {}

    return {
        "thought": thought,
        "action_description": action_desc,
        "answer": answer
    }
