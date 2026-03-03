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
from typing import Dict

import logging
import ray
import traceback
import torch

from hammer_trainer.utils.dataset.vision_utils import base64_to_image
from hammer_server.client import HammerEnvClient
from .utils.torch_functional import postprocess_data
from .utils.uitars import (
    MOBILE_USE_DOUBAO,
    SYSTEM_PROMPT,
    add_box_token,
    parse_action_to_structure_output,
    parsing_response_to_android_world_code,
)

logger = logging.getLogger(__file__)

uitars_instruction_prompt = MOBILE_USE_DOUBAO


@ray.remote(num_cpus=1)
# @ray.remote
class EnvWorker:
    instruction_prompt = uitars_instruction_prompt

    def __init__(self, worker_id, config):
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

        self.model = "uitars"
        logger.info("Start to create android env.")
        self.env_client = HammerEnvClient(src=config.env.src)
        obs = self.env_client.request_device()
        assert obs["is_alive"]
        image = base64_to_image(obs["screenshot"])
        self.origin_width, self.origin_height = image.size
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

    def init_task(self, task: Dict):
        self._is_done = False
        self.task = task
        self.steps = 0
        self._history_messages.clear()

        obs = {"goal": None, "is_alive": False, "screenshot": None, "screenshot_som": None}

        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        try:
            if (task_param_file := task.get("task_param_file")) and (goal := task.get("goal")):
                obs = self.env_client.init_task(task=task["task"], task_param_file=task_param_file)
                assert goal == obs["goal"]
            else:
                obs = self.env_client.init_task(
                    task=task["task"], task_param_file=task.get("task_param_file")
                )
                self.task["goal"] = obs["goal"]
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.instruction_prompt.format(
                                language="english", instruction=obs["goal"]
                            ),
                        }
                    ],
                }
            )

            last_msg = []
            screenshot = obs.get("screenshot")
            if screenshot:
                last_msg.append(
                    {
                        "type": "image",
                        "image": screenshot,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )
            exception = obs.get("exception")
            if exception:
                last_msg.append({"type": "text", "text": exception})
            if last_msg:
                messages.append({"role": "user", "content": last_msg})

            if obs["is_alive"]:
                self._is_alive = True
            else:
                self._is_alive = False

        except Exception as e:
            logger.error(f"task init exception: {e}\ntraceback: {traceback.format_exc()}")
            self._is_alive = False

        self._history_messages.extend(messages)
        logger.debug(f"# of messages: {len(self._history_messages)}")

        return {
            "id": self.worker_id,
            "task": self.task,
            "messages": self._history_messages,
            "is_done": self._is_done,
            "format_reward": 0.0,
            "is_alive": self._is_alive,
        }

    def step(self, action_text):
        last_msg = self._history_messages[-1]
        assert last_msg["role"] == "user"
        messages = []
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": add_box_token(action_text)}],
            }
        )
        # format
        try:
            parsed_actions = parse_action_to_structure_output(
                text=action_text,
                factor=self.action_parse_res_factor,
                origin_resized_height=self.origin_height,
                origin_resized_width=self.origin_width,
                model_type=self.model_type,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
            )

            actions = []
            for parsed_action in parsed_actions:
                action = parsing_response_to_android_world_code(
                    parsed_response=parsed_action,
                    obs_image_height=self.origin_height,
                    obs_image_width=self.origin_width,
                )
                actions.append(action)
            format_reward = 0.0
        except Exception as e:
            logger.error(
                f"action `{action_text}` parse error\nexception: {e}\ntraceback: {traceback.format_exc()}"
            )
            self._is_done = True  # error output format, stop the trajectory immediately
            format_reward = -1.0
            actions = []

        # execution
        obs = {"is_alive": False, "screenshot": None, "screenshot_som": None, "exception": None}
        try:
            for action in actions:
                obs = self.env_client.step(action)

                last_msg_content = []
                screenshot = obs.get("screenshot")
                if screenshot:
                    last_msg_content.append(
                        {
                            "type": "image",
                            "image": screenshot,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        }
                    )
                exception = obs.get("exception")
                if exception:
                    last_msg_content.append({"type": "text", "text": exception})
                if last_msg_content:
                    messages.append({"role": "user", "content": last_msg_content})

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

        self._history_messages.extend(messages)
        logger.debug(f"# of messages: {len(self._history_messages)}")

        return {
            "id": self.worker_id,
            "task": self.task,
            "messages": self._history_messages,
            "is_done": self._is_done,
            "format_reward": format_reward,
            "is_alive": self._is_alive,
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
