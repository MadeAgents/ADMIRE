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

from qwen_vl_utils.vision_process import (
    MIN_PIXELS,
    MAX_PIXELS,
    process_vision_info,
)
from verl.utils.tokenizer import hf_processor, hf_tokenizer
from verl.models.transformers.qwen2_vl import get_rope_index
from typing import Dict, Optional, Tuple
import ast
import re
import json
import os
import shutil
from datetime import datetime

import logging
import ray
import traceback
import torch
from copy import deepcopy
from qwen_vl_utils.vision_process import smart_resize, IMAGE_FACTOR

from hammer_trainer.utils.dataset.vision_utils import base64_to_image, draw_interactive_ui_elements_on_screenshot
from hammer_server.client import HammerEnvClient
from hammer_trainer.utils.torch_functional import postprocess_data
from hammer_trainer.utils.hammer_ui import (
    MOBILE_USE_HAMMER,
    SYSTEM_PROMPT,
    parse_action_to_structure_output as parse_action_to_structure_output_hammer,
)
from hammer_trainer.utils.uitars import (
    MOBILE_USE_DOUBAO,
    SYSTEM_PROMPT as UITARS_SYSTEM_PROMPT,
)
from hammer_trainer_stepwise.uitars_action_parser import (
    parse_action_to_structure_output as parse_action_to_structure_output_tars,
    add_box_token,
)
from hammer_trainer_stepwise.utils import get_validity_reward

logger = logging.getLogger(__file__)

@ray.remote(num_cpus=1)
# @ray.remote
class EnvWorker:
    def __init__(self, worker_id, config, src):
        self.worker_id = worker_id
        self.config = config

        self.tokenizer = hf_tokenizer(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.actor_rollout_ref.model.trust_remote_code,
            use_fast=True,
        )
        self.processor = hf_processor(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.actor_rollout_ref.model.trust_remote_code,
            use_fast=True,
        )

        self.model_variant = getattr(self.config.env, "model_variant", "hammer")
        if self.model_variant not in {"hammer", "tars"}:
            logger.warning(
                "Unknown model_variant `%s`, fallback to hammer.", self.model_variant
            )
            self.model_variant = "hammer"
        self.prompt_language = getattr(self.config.env, "prompt_language", "english")

        if self.model_variant == "tars":
            self.model = "tars"
            self._system_prompt_template = UITARS_SYSTEM_PROMPT
            self._instruction_prompt_template = MOBILE_USE_DOUBAO
            self._parse_action_fn = parse_action_to_structure_output_tars
        else:
            self.model = "hammer"
            self._system_prompt_template = SYSTEM_PROMPT
            self._instruction_prompt_template = MOBILE_USE_HAMMER
            self._parse_action_fn = parse_action_to_structure_output_hammer

        logger.info("Start to create android env.")
        self.env_client = HammerEnvClient(src=src)
        obs = self.env_client.request_device()
        assert obs["is_alive"]
        # image = base64_to_image(obs["screenshot"])
        # self.origin_width, self.origin_height = image.size
        self.min_pixels = self.config.env.min_pixels or MIN_PIXELS
        self.max_pixels = self.config.env.max_pixels or MAX_PIXELS

        self._is_done = False  # task is done (decided by agent) or exceeding max steps
        self._is_alive = True  # device exited abnormally or task init failed
        self.max_steps = config.env.max_steps

        self.action_parse_res_factor = 1000
        assert self.processor.__class__.__name__ in ["Qwen2_5_VLProcessor", "Qwen2VLProcessor"]
        self.model_type = (
            "qwen25vl" if self.processor.__class__.__name__ == "Qwen2_5_VLProcessor" else "qwen2vl"
        )

        self.steps = 0
        self._history_messages = []
        self.task = None
        self.is_training = False
        self.save_path = None
        self.timestamp = None

    def _build_system_prompt(self, width=None, height=None):
        if self.model_variant == "hammer" and width is not None and height is not None:
            return self._system_prompt_template.format(width=width, height=height)
        return self._system_prompt_template

    def _build_user_prompt(self, goal, history_text="None"):
        if self.model_variant == "tars":
            instruction = goal
            if history_text and history_text != "None":
                instruction = f"{goal}\n\nRecent steps:\n{history_text}"
            return self._instruction_prompt_template.format(
                language=self.prompt_language,
                instruction=instruction,
            )
        history = history_text if history_text else "None"
        return self._instruction_prompt_template.format(goal=goal, history=history)

    def _format_history_text(self, history_infos):
        if not history_infos.get("action_description"):
            return "None"
        descs = history_infos["action_description"]
        if self.config.env.max_history_descs is not None:
            descs = descs[-self.config.env.max_history_descs :]
            offset = len(history_infos["action_description"]) - len(descs)
        else:
            offset = 0
        formatted = [f"Step {offset + idx + 1}: {desc}" for idx, desc in enumerate(descs)]
        return "; ".join(formatted) if formatted else "None"

    @staticmethod
    def _parse_box(box_str: Optional[str]) -> Optional[list]:
        if box_str is None:
            return None
        try:
            cleaned = str(box_str)
            point_match = re.search(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", cleaned)
            if point_match:
                cleaned = f"({point_match.group(1)},{point_match.group(2)})"
            point_match2 = re.search(r"<point\((\d+)\s*,\s*(\d+)\)>", cleaned)
            if point_match2:
                cleaned = f"({point_match2.group(1)},{point_match2.group(2)})"
            cleaned = cleaned.replace("<|box_start|>", "").replace("<|box_end|>", "")
            cleaned = cleaned.replace("<point>", "").replace("</point>", "")
            coords = ast.literal_eval(cleaned)
        except Exception:
            return None
        if isinstance(coords, (list, tuple)):
            return list(coords)
        return None

    @staticmethod
    def _convert_coord(value: float, size: int) -> int:
        if value is None:
            return 0
        if abs(value) <= 2:  # normalized
            value = value * size
        value = max(0, min(value, size - 1))
        return int(round(value))

    def _box_to_point(self, box_str: Optional[str], width: int, height: int) -> Optional[list]:
        coords = self._parse_box(box_str)
        if not coords:
            return None
        if len(coords) >= 4:
            x = (float(coords[0]) + float(coords[2])) / 2
            y = (float(coords[1]) + float(coords[3])) / 2
        elif len(coords) >= 2:
            x = float(coords[0])
            y = float(coords[1])
        else:
            return None
        return [self._convert_coord(x, width), self._convert_coord(y, height)]

    def _box_pair_to_points(
        self, start_box: Optional[str], end_box: Optional[str], width: int, height: int
    ) -> Tuple[Optional[list], Optional[list]]:
        start = self._box_to_point(start_box, width, height)
        end = self._box_to_point(end_box, width, height)
        return start, end

    def _map_tars_action(
        self, parsed_action: Dict[str, any], width: int, height: int
    ) -> Optional[Tuple[str, Dict[str, any], str]]:
        action_type = (parsed_action.get("action_type") or "").lower()
        inputs = parsed_action.get("action_inputs", {}) or {}
        desc = parsed_action.get("text") or action_type

        def point_from(key):
            return self._box_to_point(inputs.get(key), width, height)

        hammer_type = "unknown"
        action_args: Dict[str, any] = {}

        def assign_xy(coord: Optional[list]):
            if coord:
                action_args["x"] = coord[0]
                action_args["y"] = coord[1]

        if action_type in {"click", "left_single", "hover"}:
            hammer_type = "click"
            assign_xy(point_from("start_box"))
        elif action_type in {"left_double", "double_tap"}:
            hammer_type = "double_tap"
            assign_xy(point_from("start_box"))
        elif action_type == "long_press":
            hammer_type = "long_press"
            assign_xy(point_from("start_box"))
        elif action_type in {"drag", "select", "swipe"}:
            hammer_type = "swipe"
            start, end = self._box_pair_to_points(
                inputs.get("start_box"), inputs.get("end_box"), width, height
            )
            if start:
                action_args["x"] = start[0]
                action_args["y"] = start[1]
            if end:
                action_args["x2"] = end[0]
                action_args["y2"] = end[1]
        elif action_type == "scroll":
            hammer_type = "scroll"
            assign_xy(point_from("start_box"))
            action_args["direction"] = inputs.get("direction", "down")
        elif action_type in {"type", "input_text"}:
            hammer_type = "input_text"
            action_args["text"] = inputs.get("content", "")
            assign_xy(point_from("start_box"))
        elif action_type == "open_app":
            hammer_type = "open_app"
            action_args["app_name"] = inputs.get("app_name") or inputs.get("content", "")
        elif action_type in {"press_back", "press_return"}:
            hammer_type = "navigate_back"
        elif action_type == "press_home":
            hammer_type = "navigate_home"
        elif action_type == "press_keyboard":
            hammer_type = "keyboard_enter"
        elif action_type in {"finished", "status"}:
            hammer_type = "status"
            action_args["goal_status"] = inputs.get("content", "success")
        elif action_type == "wait":
            hammer_type = "wait"
        elif action_type == "clear_text":
            hammer_type = "clear_text"
        elif action_type == "answer":
            hammer_type = "answer"
            action_args["text"] = inputs.get("content", "")

        return hammer_type, action_args, desc

    def load_content(self, content):
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"

        raise ValueError(f"Unknown content type: {content}")

    def messages_to_input_tensors(self, messages):
        processor = self.processor
        images, _ = process_vision_info(messages)

        input_ids = []
        labels = []
        attention_mask = []

        image_idx = 0
        pixel_values = []
        image_grid_thw = []
        for msg in messages:
            role = msg["role"]
            content = self.load_content(msg["content"])
            prompt = f"<|im_start|>{role}\n" + content + "<|im_end|>\n"

            num_image_msg = prompt.count("<|image_pad|>")
            if num_image_msg > 0:
                result = processor(
                    images=images[image_idx : image_idx + num_image_msg],
                    text=[prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                image_idx += num_image_msg
            else:
                result = processor(
                    images=None, text=[prompt], add_special_tokens=False, return_tensors="pt"
                )

            msg_input_ids = result.pop("input_ids")[0]
            msg_attention_mask = result.pop("attention_mask")[0]
            if "pixel_values" in result:  # 10764, 1176
                pixel_values.append(result["pixel_values"])
            if "image_grid_thw" in result:
                image_grid_thw.append(result["image_grid_thw"])

            input_ids.append(msg_input_ids)
            attention_mask.append(msg_attention_mask)
            if role in ["system", "user"]:
                labels.append(torch.full_like(msg_input_ids, -100))
            else:
                labels.append(msg_input_ids)

        input_ids = torch.cat(input_ids, dim=0)
        labels.append(torch.tensor([-100]))
        labels = torch.cat(labels, dim=0)[1:]
        attention_mask = torch.cat(attention_mask, dim=0)
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

        return input_ids, labels, attention_mask, pixel_values, image_grid_thw

    def input_tensors(self):
        input_ids, labels, attention_mask, pixel_values, image_grid_thw = (
            self.messages_to_input_tensors(self._history_messages)
        )

        position_ids = get_rope_index(
            processor=self.processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        input_ids, attention_mask, position_ids, labels = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
            labels=labels,
        )
        data = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            multi_modal_inputs = dict()
            multi_modal_inputs["pixel_values"] = pixel_values
            multi_modal_inputs["image_grid_thw"] = image_grid_thw
            data["multi_modal_inputs"] = multi_modal_inputs
        return data

    def init_task(self, task: Dict, is_training: bool):
        self._is_done = False
        self.task = task
        self.steps = 0
        self._history_messages.clear()
        self.is_training = is_training
        self.timestamp = datetime.now().strftime("%m%d%H%M%S")

        logger.info(f"Init task: {self.task['task']}")

        if self.is_training:
            self.save_path = self.config.trainer.get("rollout_data_dir", None)
        else:
            self.save_path = self.config.trainer.get("validation_data_dir", None)
        if self.save_path:
            os.makedirs(f"{self.save_path}/imgs", exist_ok=True)
            os.makedirs(f"{self.save_path}/history_infos", exist_ok=True)
        image_path = None
        error_info = ""

        obs = {"goal": None, "is_alive": False, "screenshot": None, "screenshot_som": None, "ui_elements": None}

        try:
            if (task_param_file := task.get("task_param_file")) and (goal := task.get("goal")):
                obs = self.env_client.init_task(task=task["task"], task_param_file=task_param_file)
                assert goal == obs["goal"]
            else:
                obs = self.env_client.init_task(
                    task=task["task"], task_param_file=task.get("task_param_file")
                )
                self.task["goal"] = obs["goal"]
            
            last_msg = []
            screenshot = obs.get("screenshot")
            ui_elements = obs.get("ui_elements")
            if screenshot:
                screenshot_decoded = base64_to_image(screenshot)
                if self.config.env.draw_interactive_ui_elements:
                    screenshot_decoded = draw_interactive_ui_elements_on_screenshot(screenshot_decoded, ui_elements)
                screen_size = screenshot_decoded.size
                smart_resize_width, smart_resize_height = screen_size
                if self.model_type == "qwen25vl":
                    smart_resize_height, smart_resize_width = smart_resize(
                        screen_size[1],
                        screen_size[0],
                        factor=IMAGE_FACTOR,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                system_prompt_text = self._build_system_prompt(
                    width=smart_resize_width, height=smart_resize_height
                )

                last_msg.append(
                    {
                        "type": "image",
                        "image": screenshot,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )

                if self.save_path:
                    image_path = f"{self.save_path}/imgs/{self.task['task']}-{self.worker_id}-{self.timestamp}-init.png"
                    screenshot_decoded.save(image_path)
            else:
                system_prompt_text = self._build_system_prompt()
            
            user_prompt_text = self._build_user_prompt(goal=obs["goal"], history_text="None")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt_text}]},
            ]
            exception = obs.get("exception")
            if exception:
                exception = str(exception).split('\n')
                exception = exception[0].strip() if len(exception) == 0 else exception[0].strip() + '\n' + exception[-1].strip()
                last_msg.append({"type": "text", "text": exception})
            if last_msg:
                messages[-1]["content"].extend(last_msg)

            self._is_alive = bool(obs["is_alive"])

        except Exception as e:
            logger.error(f"task init exception: {e}\ntraceback: {traceback.format_exc()}")
            self._is_alive = False
            error_info = f"task init exception: {e}"
            system_prompt_text = self._build_system_prompt()
            user_prompt_text = self._build_user_prompt(goal=self.task.get("goal", ""))
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt_text}]},
            ]

        self._history_messages.extend(messages)
        logger.debug(f"# of messages: {len(self._history_messages)}")

        return {
            "id": self.worker_id,
            "task": self.task,
            "task_id": self.task["task_id"],
            "goal": self.task["goal"], # Expose goal for the judge
            "messages": self._history_messages,
            "is_done": self._is_done,
            "format_reward": 0.0,
            "validity_reward": 0.0,
            "is_alive": self._is_alive,
            "history_infos": {"action_text": [], "action_description": [], "action": [], "image_path": [image_path], "error_info": [error_info], "format_reward": [], "validity_reward": [], "goal": self.task["goal"], "task": self.task["task"]},
        }

    def step(self, action_text, history_infos):
        messages = []
        try:
            last_msg = self._history_messages[-1]
            assert last_msg["role"] == "user"
            try:
                last_obs = last_msg["content"][1]["image"]
            except:
                x = [x for _msg in self._history_messages for x in _msg["content"] if "image" in x][-1]
                last_obs = x["image"]
            image = base64_to_image(last_obs)
            origin_width, origin_height = image.size
            assistant_text = action_text
            if self.model_variant == "tars":
                assistant_text = add_box_token(action_text)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )
            # format
            parsed_actions = self._parse_action_fn(
                text=action_text,
                factor=self.action_parse_res_factor,
                origin_resized_height=origin_height,
                origin_resized_width=origin_width,
                model_type=self.model_type,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
            )

            actions = []
            action_descs = []
            for parsed_action in parsed_actions:
                if self.model_variant == "tars":
                    hammer_type, hammer_args, action_desc = self._map_tars_action(
                        parsed_action, origin_width, origin_height
                    )
                    action = {
                        "name": hammer_type,
                        "arguments": json.dumps(hammer_args, ensure_ascii=False),
                    }
                else:
                    action = deepcopy(parsed_action["action"])
                    action_type = action.pop("action_type")
                    hammer_args = action
                    action_desc = parsed_action.get("action_description", action_type)
                    action = {
                        "name": action_type,
                        "arguments": json.dumps(hammer_args, ensure_ascii=False),
                    }
                actions.append(action)
                action_descs.append(action_desc)
            format_reward = 0.0
            error_info = ""
        except Exception as e:
            logger.error(
                f"action `{action_text}` parse error\nexception: {e}\ntraceback: {traceback.format_exc()}"
            )
            self._is_done = True  # error output format, stop the trajectory immediately
            format_reward = -1.0
            actions = []
            action_descs = []
            error_info = f"action `{action_text}` parse error\nexception: {e}"
        history_infos["action_text"].append(action_text)
        history_infos["action_description"].extend(action_descs[: 1])
        if len(history_infos["action_description"]) != len(history_infos["action_text"]):
            history_infos["action_description"].append("")
        history_infos["action"].extend(actions[: 1])
        if len(history_infos["action"]) != len(history_infos["action_text"]):
            history_infos["action"].append("")
        history_infos["format_reward"].append(format_reward)
        image_path = None

        # execution
        obs = {"is_alive": False, "screenshot": None, "screenshot_som": None, "exception": None, "ui_elements": None}
        validity_reward = 0
        try:
            for action in actions[: 1]:
                obs = self.env_client.step(action)

                last_msg_content = []
                screenshot = obs.get("screenshot")
                ui_elements = obs.get("ui_elements")
                if ui_elements and self.config.strategy.use_validity_reward:
                    validity_reward = get_validity_reward(action, ui_elements)
                if screenshot:
                    screenshot_decoded = base64_to_image(screenshot)
                    if self.config.env.draw_interactive_ui_elements:
                        screenshot_decoded = draw_interactive_ui_elements_on_screenshot(screenshot_decoded, ui_elements)
                    screen_size = screenshot_decoded.size
                    if self.model_type == "qwen25vl":
                        smart_resize_height, smart_resize_width = smart_resize(
                            screen_size[1],
                            screen_size[0],
                            factor=IMAGE_FACTOR,
                            min_pixels=self.min_pixels,
                            max_pixels=self.max_pixels,
                        )
                    hist_text = self._format_history_text(history_infos)
                    messages.extend([
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self._build_system_prompt(
                                        width=smart_resize_width, height=smart_resize_height
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self._build_user_prompt(
                                        goal=self.task["goal"], history_text=hist_text
                                    ),
                                }
                            ],
                        },
                    ])

                    last_msg_content.append(
                        {
                            "type": "image",
                            "image": screenshot,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        }
                    )

                    if self.save_path:
                        image_path = f"{self.save_path}/imgs/{self.task['task']}-{self.worker_id}-{self.timestamp}-{self.steps}.png"
                        screenshot_decoded.save(image_path)
                exception = obs.get("exception")
                if exception:
                    exception = str(exception).split('\n')
                    exception = exception[0].strip() if len(exception) == 0 else exception[0].strip() + '\n' + exception[-1].strip()
                    last_msg_content.append({"type": "text", "text": exception})
                if last_msg_content:
                    if messages[-1]["role"] != "user":
                        messages.append({"role": "user", "content": last_msg_content})
                    else:
                        messages[-1]["content"].extend(last_msg_content)

                # device exited abnormally
                if not obs["is_alive"]:
                    self._is_alive = False
                    break

                self.steps += 1
                # finished
                if action["name"] == "status":
                    self._is_done = True
                    break
                # terminated
                if self.steps >= self.max_steps:
                    self._is_done = True
                    break

        except Exception as e:
            logger.error(f"task step exception: {e}\ntraceback: {traceback.format_exc()}")
            self._is_alive = False
            error_info = (f"FIRST {error_info}\nSECOND task step exception: {e}" if error_info else f"task step exception: {e}")

        history_infos["image_path"].append(image_path)
        history_infos["error_info"].append(error_info)
        history_infos["validity_reward"].append(validity_reward)
        self._history_messages.extend(messages)
        logger.debug(f"# of messages: {len(self._history_messages)}")

        if self.save_path:
            with open(f"{self.save_path}/history_infos/{self.task['task']}-{self.worker_id}-{self.timestamp}.json", 'w') as outfile:
                json.dump(history_infos | {
                    "goal": self.task["goal"], 
                    "task": self.task["task"],
                    "steps": self.steps, 
                    "is_alive": self._is_alive,
                    "is_done": self._is_done
                }, outfile, indent=2, ensure_ascii=False)
        return {
            "id": self.worker_id,
            "task": self.task,
            "task_id": self.task["task_id"],
            "goal": self.task["goal"],
            "messages": self._history_messages,
            "is_done": self._is_done,
            "format_reward": format_reward,
            "validity_reward": validity_reward,
            "is_alive": self._is_alive,
            "history_infos": history_infos
        }

    def evaluate(self):
        try:
            obs = self.env_client.submit()
            if not obs["is_alive"]:
                self._is_alive = False
            return 1.0 if obs["is_successful"] == "任务成功 ✅" else 0.0
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return 0.0

    def release(self):
        self.env_client.close()

    def history_messages(self):
        return self._history_messages

    def is_done(self):
        return self._is_done

    def is_alive(self):
        return self._is_alive

    def reset(self):
        self._is_alive = True
        self._is_done = False
        self.task = None
        self._history_messages.clear()
