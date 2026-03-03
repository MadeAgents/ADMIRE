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

import re
import tempfile
from PIL import Image, ImageDraw
from absl import logging
from copy import deepcopy
from pathlib import Path

import gradio as gr
import json
import math

from hammer_server.utils import (
    base64_to_image,
    get_action_param_prompt,
    get_action_types,
    get_ip,
    image_to_base64,
)

logging.set_verbosity("debug")

SERV_DIR = Path(__file__).resolve().parent
with open(SERV_DIR / "css/block.css", "r") as f:
    block_css = f.read()


class State:
    def __init__(self):
        self.trajectory = []
        self.task = None
        self.goal = None
        self.is_successful = None

    def load(self, filepath):
        with open(filepath, mode="r") as f:
            record = json.load(f)
        self.task = record["task"]
        self.goal = record["goal"]
        self.is_successful = record["success"]
        self.trajectory = record["trajectory"]

        logging.info(
            f"task: {self.task}\ngoal: {self.goal}\nis_successful: {self.is_successful}\n# of steps: {self.num_steps}"
        )

    @property
    def num_steps(self):
        return len(self.trajectory)

    def to_chatbot(self):
        chatbot = []
        for idx, step in enumerate(self.trajectory):
            response = step["response"]
            action = step["action"]
            screenshot = process_image(
                step["observation"]["screenshot_som"], action=json.loads(action)
            )
            msg_pair = (
                f"step {idx}",
                f"""<div><p>response: {response}</p><p>action: {action}</p></div><div class="image-container"><img src="{screenshot}" /></div>""",
            )
            chatbot.append(msg_pair)
        return chatbot

    def dumps(self):
        return json.dumps(
            {
                "task": self.task,
                "goal": self.goal,
                "trajectory": self.trajectory,
                "success": self.is_successful,
            },
            ensure_ascii=False,
            indent=4,
        )

    def update(self, step: int, action_type, action_args):
        assert (step >= 0) and (step < self.num_steps)
        action = _parse_action(action_type=action_type, action_args=action_args)

        self.trajectory = self.trajectory[: step + 1]
        last_step = self.trajectory[-1]
        last_step["action"] = json.dumps(action)
        last_step["revised"] = True


def mark_action(image, action):
    """
    action (android_world/android_world/env/actuation.py):

    # scroll direction
    x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)
    start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    if direction == 'down':
        end_x, end_y = (x_min + x_max) // 2, y_min
    elif direction == 'up':
        end_x, end_y = (x_min + x_max) // 2, y_max
    elif direction == 'right':
        end_x, end_y = x_min, (y_min + y_max) // 2
    elif direction == 'left':
        end_x, end_y = x_max, (y_min + y_max) // 2

    # swipe direction
    mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
    if direction == 'down':
        start_x, start_y = mid_x, 0
        end_x, end_y = mid_x, screen_height
    elif direction == 'up':
        start_x, start_y = mid_x, screen_height
        end_x, end_y = mid_x, 0
    elif direction == 'left':
        start_x, start_y = 0, mid_y
        end_x, end_y = screen_width, mid_y
    elif direction == 'right':
        start_x, start_y = screen_width, mid_y
        end_x, end_y = 0, mid_y
    """

    def _plot_point(draw, point, radius, fill="green"):
        draw.ellipse(
            [
                (point[0] - radius, point[1] - radius),
                (point[0] + radius, point[1] + radius),
            ],
            fill=fill,
        )

    def _plot_arrow(draw, start, end, fill="green", arrow_size=10, width=2):
        draw.line([start, end], fill=fill, width=width)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        theta = -math.atan2(dy, dx)
        angle = math.radians(30)
        adj_angle1 = theta + angle
        adj_angle2 = theta - angle

        point1 = (
            end[0] - arrow_size * math.cos(adj_angle1),
            end[1] + arrow_size * math.sin(adj_angle1),
        )
        point2 = (
            end[0] - arrow_size * math.cos(adj_angle2),
            end[1] + arrow_size * math.sin(adj_angle2),
        )
        draw.polygon([end, point1, point2], fill=fill)

    w, h = image.size
    radius = w * 0.02
    arrow_size = w * 0.05
    line_width = int(w * 0.01)
    draw = ImageDraw.Draw(image)

    action = deepcopy(action)

    action_type = action.pop("action_type")
    action_args = action

    if action_type in ["click", "double_tap", "long_press"]:
        x, y = action_args.get("x"), action_args.get("y")
        if x is not None and y is not None:
            _plot_point(draw=draw, point=(x, y), radius=radius)
    elif action_type == "input_text":
        x, y = action_args.get("x"), action_args.get("y")
        if x is not None and y is not None:
            _plot_point(draw=draw, point=(x, y), radius=radius)
    elif action_type == "scroll":
        direction = action_args["direction"]
        x_min, y_min, x_max, y_max = (0, 0, w, h)
        start = (x_min + x_max) // 2, (y_min + y_max) // 2
        if direction == "down":
            end = (x_min + x_max) // 2, y_min + arrow_size
        elif direction == "up":
            end = (x_min + x_max) // 2, y_max - arrow_size
        elif direction == "right":
            end = x_min, (y_min + y_max) // 2
        elif direction == "left":
            end = x_max, (y_min + y_max) // 2
        _plot_arrow(draw=draw, start=start, end=end, arrow_size=arrow_size, width=line_width)
    elif action_type == "swipe":
        direction = action_args["direction"]
        mid_x, mid_y = 0.5 * w, 0.5 * h
        if direction == "down":
            start = mid_x, 0
            end = mid_x, h
        elif direction == "up":
            start = mid_x, h
            end = mid_x, 0
        elif direction == "left":
            start = 0, mid_y
            end = w, mid_y
        elif direction == "right":
            start = w, mid_y
            end = 0, mid_y
        _plot_arrow(draw=draw, start=start, end=end, arrow_size=arrow_size, width=line_width)
    else:
        pass


def process_image(image_base64, action=None, width=240):

    image = base64_to_image(image_base64=image_base64)
    if action is not None:
        mark_action(image=image, action=action)
    w, h = image.size
    ratio = width / w
    height = int(ratio * h)
    image = image.resize(size=(width, height), resample=Image.Resampling.LANCZOS)
    image_base64 = image_to_base64(image)
    return image_base64


def update_action_param_prompt(action_type, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"update_action_param_prompt. ip: {ip}")

    prompt = get_action_param_prompt(action_type=action_type)
    return gr.update(value=prompt, submit_btn=True, interactive=True)


def upload(_state: State, _file, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"upload. ip: {ip}")

    if _state is None:
        _state = State()
    _state.load(_file)
    _chatbot = _state.to_chatbot()
    num_steps = _state.num_steps
    step_selector = [f"step_{idx}" for idx in range(num_steps)]

    return (
        _state,
        _chatbot,
        gr.update(value=None, choices=step_selector, interactive=True),
    )


def select_step(request: gr.Request):
    ip = get_ip(request)
    logging.info(f"select_step. ip: {ip}")

    return gr.update(interactive=True)  # action_type_selector


def _parse_action(action_type, action_args):
    action = {"action_type": action_type}
    # action arguments
    pattern = r"```json\s*({.*?})\s*```"
    match = re.search(pattern, action_args, re.DOTALL)
    if match:
        action_args = match.group(1)
    try:
        action_args = json.loads(action_args)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        return
    action.update(action_args)
    return action


def update_state(_state: State, _step, _action_type, _action_args, request: gr.Request):
    ip = get_ip(request)
    logging.info(
        f"update_state. ip: {ip}, step: {_step}, action_type: {_action_type}, action_args: {_action_args}"
    )

    _state.update(step=_step, action_type=_action_type, action_args=_action_args)
    chatbot = _state.to_chatbot()
    return (
        _state,
        chatbot,
        gr.update(interactive=True),  # download_btn
    )


def download(_state: State, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"download. ip: {ip}")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".json") as f:
        f.write(_state.dumps())
    return f.name


def _build_demo():
    notice_markdown = f"""# 🤖 GUI 智能体操作轨迹人工校正"""
    state = gr.State()

    with gr.Blocks(css=block_css) as demo:
        gr.Markdown(notice_markdown, elem_id="notice_markdown")
        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    value=[(None, "hi")],
                    type="tuples",
                    label="Android Device",
                    height=900,
                    resizable=True,
                    show_copy_button=True,
                    show_copy_all_button=True,
                    # editable="all",
                )
                step_selector = gr.Radio(choices=[], type="index", interactive=False)
                action_type_selector = gr.Radio(
                    choices=get_action_types(),
                    label="Next Action Type",
                    interactive=False,
                )
                action_param_textbox = gr.Textbox(
                    lines=10,
                    label="Next Action Parameters",
                    submit_btn=False,
                    interactive=False,
                )
                upload_btn = gr.File(label="选择文件", file_count="single", file_types=[".json"])
                download_btn = gr.DownloadButton(
                    label="💾 Download (下载)", variant="primary", interactive=False
                )

        state = gr.State()

        upload_btn.upload(
            fn=upload,
            inputs=[state, upload_btn],
            outputs=[state, chatbot, step_selector],
        )
        step_selector.select(fn=select_step, inputs=None, outputs=action_type_selector)
        action_type_selector.change(
            fn=update_action_param_prompt,
            inputs=[action_type_selector],
            outputs=[action_param_textbox],
        )
        action_param_textbox.submit(
            fn=update_state,
            inputs=[state, step_selector, action_type_selector, action_param_textbox],
            outputs=[state, chatbot, download_btn],
        )
        download_btn.click(fn=download, inputs=state, outputs=download_btn)
    return demo


def main():
    demo = _build_demo()
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
