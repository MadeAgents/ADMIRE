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

import json
from typing import Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent
from copy import deepcopy
from PIL import Image
import numpy as np
import random

from qwen_vl_utils.vision_process import process_vision_info
from typing import Any, Dict, List, Optional, Union
from verl.protocol import DataProto

from android_world.env.representation_utils import UIElement
from openai import OpenAI
from hammer_server.utils import image_to_base64
from hammer_trainer.utils.hammer_ui import extract_info_from_action_text, OUTPUT_FORMAT


def pad_dataproto_to_constant_size(data: "DataProto", min_size: int):
    assert isinstance(data, DataProto), "data must be a DataProto"

    for i in range(11):
        s = 2 ** i
        if s >= min_size and s >= len(data):
            size = s
            break

    if len(data) < size:
        if len(data) == 0:
            print("padding a DataProto with no item, no changed made")
            return data, 0
        pad_size = size - len(data)
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def diff_image(
    img1: Image.Image,
    img2: Image.Image,
    pixel_threshold: int = 5,
    area_threshold: int = 1000,
    max_boxes: int = 2,
    merge_threshold: int = 20,
):
    import cv2
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    opencv_image1 = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
    opencv_image2 = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)

    diff = cv2.absdiff(opencv_image1, opencv_image2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, pixel_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((merge_threshold, merge_threshold), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"DIFF IMAGE: Number of raw contours: {len(contours)}")
    contours = [c for c in contours if cv2.contourArea(c) > area_threshold]
    logger.info(f"DIFF IMAGE: Number of filtered contours: {len(contours)}")
    if len(contours) == 0:
        logger.info("DIFF IMAGE: No contours found.")
        return None, None
    if len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
        if (x, y, w, h) == (0, 0, img2.size[0], img2.size[1]):
            logger.info("DIFF IMAGE: The two images are exactly different.")
            return None, None
    if len(contours) > max_boxes:
        logger.info(f"DIFF IMAGE: Too many contours found: {len(contours)}")
        return None, None

    new_img1 = opencv_image1.copy()
    new_img2 = opencv_image2.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x-3)
        y = max(0, y-3)
        w = min(w + 6, img1.size[0] - x)
        h = min(h + 6, img1.size[1] - y)
        cv2.rectangle(new_img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.rectangle(new_img2, (x, y), (x + w, y + h), (0, 0, 255), 3)
    new_img1 = Image.fromarray(cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB))
    new_img2 = Image.fromarray(cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB))
    return new_img1, new_img2


def get_validity_reward(action: Dict, ui_elements: List[UIElement]):
    action_type = action["name"]
    action_args = json.loads(action["arguments"])

    if action_type == "click" or action_type == "long_press":
        x = action_args["x"]
        y = action_args["y"]
        is_enabled = False
        for elem in ui_elements:
            x_min, x_max, y_min, y_max = elem.x_min, elem.x_max, elem.y_min, elem.y_max
            if x_min <= x <= x_max and y_min <= y <= y_max and elem.is_enabled:
                is_enabled = True
        return 0 if is_enabled else -1
    elif action_type == "input_text":
        is_focused = False
        for elem in ui_elements:
            if elem.is_focused:
                if elem.is_editable:
                    is_focused = True
                if 'EditText' in elem.class_name or 'AutoCompleteTextView' in elem.class_name or 'MultiAutoCompleteTextView' in elem.class_name or 'SearchView' in elem.class_name:
                    is_focused = True
        return 0 if is_focused else -1


def get_batch_human_helps(
    batch_messages, 
    task_ids: List[str],
    human_helps_ratio: float,
    openai_api_key: str,
    openai_api_base: str,
    helper_model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
):
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    selected_idxs = []
    for task_id in set(task_ids):
        idxs = [i for i, v in enumerate(task_ids) if v == task_id]
        pick_num = int(len(idxs) * human_helps_ratio)
        selected_idxs += random.sample(idxs, pick_num)
    
    def get_human_helps(i):
        messages = batch_messages[i]
        _messages = deepcopy(messages)
        
        if i not in selected_idxs:
            return [i, _messages]

        try:
            # process images
            images, _ = process_vision_info(messages, return_video_kwargs=False)
            for j in range(len(messages)):
                if messages[j]["role"] == "user":
                    for k in range(len(messages[j]["content"])):
                        if messages[j]["content"][k]["type"] == "image":
                            messages[j]["content"][k] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_to_base64(images.pop(0))
                                }
                            }

            helper_response = openai_client.chat.completions.create(
                model=helper_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            ).choices[0].message.content
        except Exception as e:
            print(e)
            helper_response = "None"

        action_info = extract_info_from_action_text(helper_response)
        if not action_info["answer"]:
            helper_response = "None"
        else:
            helper_response = "{thought} {action_desc} So the next action might be {answer}".format(thought=action_info["thought"], action_desc=action_info["action_description"], answer=json.dumps(action_info["answer"]))
        print(helper_response)
        _messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": f"# Hints from human\n{helper_response}\n\n{OUTPUT_FORMAT}"}]
        })
        return [i, _messages]

    result = []
    with ThreadPoolExecutor(max_workers=min(len(batch_messages), 64)) as executor:
        futures = [executor.submit(get_human_helps, idx) for idx in range(len(batch_messages))]
        for future in concurrent.futures.as_completed(futures):
            result.append(future.result())
    
    batch_messages = [v for i, v in sorted(result, key=lambda x: x[0])]
    return batch_messages


# Fix Picture sequence inconsistency problem in vllm0.7.2 
# If you are using QwenAPI from 'dashscope.aliyuncs.com', replace IMAGE_PLACEHOLDER with ''
IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'

def get_batch_reflection(
    batch_messages,
    batch_history_messages, 
    batch_goals,
    openai_api_key: str,
    openai_api_base: str,
    teacher_model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    
    def get_reflection(i):
        student_messages = batch_messages[i]
        history_messages = batch_history_messages[i][-5:]
        goal = batch_goals[i]

        if [x["role"] for x in history_messages] != ["system", "user", "assistant", "system", "user"]:
            return [i, student_messages]

        try:
            # process images
            images, _ = process_vision_info(history_messages, return_video_kwargs=False)
            if len(images) != 2:
                return [i, student_messages]

            screenshot_bef, screenshot_aft = images
            action_info = extract_info_from_action_text(history_messages[2]["content"][0]["text"])
            resized_width, resized_height = screenshot_aft.size

            diff_flag = False
            new_img1, new_img2 = diff_image(screenshot_bef, screenshot_aft)
            if new_img1 is not None:
                screenshot_bef, screenshot_aft = new_img1, new_img2
                diff_flag = True

            teacher_messages = []

            # Add system prompt
            teacher_messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the latest action produced the expected behavior."
                    }
                ]
            })

            # Add user prompt
            prompt = ""
            prompt += "### User Instruction ###\n"
            prompt += f"{goal}\n\n"

            prompt += "---\n"
            prompt += f"Screenshot before latest action: {IMAGE_PLACEHOLDER}\n"
            prompt += f"Screenshot after latest action: {IMAGE_PLACEHOLDER}\n"
            prompt += f"The two images are two phone screenshots before and after your latest action. " 
            prompt += f"The width and height are {resized_width} and {resized_height} pixels, respectively.\n"
            if diff_flag:
                logger.info("The last action successfully produces some changes. The difference between the two images is highlighted in red boxes.")
                prompt += "The last action successfully produces some observable changes. The difference between the two images is highlighted in red boxes. You can find it on the images.\n"
            prompt += "\n"

            prompt += "---\n"
            prompt += "### Latest Action ###\n"
            prompt += f"Action: {json.dumps(action_info['answer'])}\n"
            prompt += f"Expectation: {action_info['action_description']}\n\n"

            prompt += "---\n"
            prompt += "Carefully examine the information provided above to determine whether the last action meets the expectation. If not, identify the failure mode and provide reasoning on the potential reason causing this failure. Note that for the “Swipe” action, it may take multiple attempts to display the expected content. Thus, for a \"Swipe\" action, if the screen shows new content, it usually meets the expectation.\n\n"

            prompt += "Provide your output in the following format containing three parts:\n\n"
            prompt += "### Outcome ###\n"
            prompt += "Choose from the following options. Give your answer as \"A\", \"B\",\"C\" or \"D\":\n"
            prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation, or on the right path to meet the expectation.\n"
            prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
            prompt += "C: Failed. The last action produces no changes.\n"
            prompt += "D: Uncertain. Can't determine whether the last action meets the expectation.\n"
            prompt += "NOTE: In some cases, the action may not produce any observable feedback, such as click a `save` or `add` button. You can't determine whether the action meets the expectation. In this case, you can choose \"D\".\n"
            prompt += "\n"

            prompt += "### Error Description ###\n"
            prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

            teacher_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_base64(screenshot_bef)}},
                    {"type": "image_url", "image_url": {"url": image_to_base64(screenshot_aft)}}
                ]
            })

            teacher_response = openai_client.chat.completions.create(
                model=teacher_model_name,
                messages=teacher_messages,
                max_tokens=max_tokens,
                temperature=temperature
            ).choices[0].message.content

            # outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
            # error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()

        except Exception as e:
            print(e)
            teacher_response = ""

        student_messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": teacher_response}]
        })
        return [i, student_messages]

    get_reflection(0)
    result = []
    with ThreadPoolExecutor(max_workers=min(len(batch_messages), 64)) as executor:
        futures = [executor.submit(get_reflection, idx) for idx in range(len(batch_messages))]
        for future in concurrent.futures.as_completed(futures):
            result.append(future.result())
    
    batch_messages = [v for i, v in sorted(result, key=lambda x: x[0])]
    return batch_messages