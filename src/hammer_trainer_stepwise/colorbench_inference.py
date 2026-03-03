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
import os 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm 
import time 
from openai import OpenAI
import base64
import random
import math
from PIL import Image
from io import BytesIO
import io
APPS = ['百度地图', '懂车帝', '中国电信', '饿了么', '番茄小说', '拼多多', '去哪儿旅行', '微信', '大麦', '美团', '抖音', '淘宝', '百度', '菜鸟', 'QQ音乐', '顺丰速运', '去哪儿', '中国联通', '小红书', '美柚', '高德地图', '微博', '哔哩哔哩', '今日头条极速版', '网易云音乐', '腾讯视频', '中国移动', '爱奇艺', '剪映', '汽车之家', '携程', 'QQ阅读', 'QQ', 'WPS Office', '58同城', 'bilibili', 'Bilibili', 'qq阅读', '京东', '今日头条', '去哪儿网', '铁路12306', '大众点评', '携程旅行', '腾讯会议', 'iQIYI', '番茄免费小说', '腾讯地图', 'OPPO商城', '软件商店', '小布助手', '支付宝', '起点读书', '红果免费短剧', '相册', '醒图', '快影', '豆包', '便签', '设置', 'DeepSeek', '即梦AI', '美图秀秀', '浏览器', '腾讯元宝', '酷狗音乐', '优酷', '知乎', '优酷视频', '快手', '瑞幸咖啡', '飞猪旅行', '天气', '日历', '计算器', '笔记', '夸克', '转转', '闲鱼', '豆包']


# openai_api_key = "empty"
# openai_api_base = "YOUR_API_BASE_URL"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


# clients = [
#     client,
#     # client1
# ]

def postprocess(vlm_response):
    vlm_response = eval(vlm_response)
    thinking = vlm_response.get('Thinking')
    action = vlm_response.get('Next Action')
    target_app_name = ""
    if ('处于' in thinking or '位于' in thinking or '处在' in thinking) and '手机' in thinking and ('主界面' in thinking or '主屏幕' in thinking or '桌面' in thinking):
        if 'CLICK' in action:
            for app in APPS:
                if app in thinking:
                    target_app_name = app
                    break 
            if target_app_name:
                vlm_response.update({'Next Action': 'OPEN' + str([target_app_name])})
                return str(vlm_response)
    if 'CLICK' in action:
        for app in APPS:
            target_str1 = '点击' + app + '图标'
            target_str2 = '点击' + app + '应用图标'
            if target_str1 in thinking or target_str2 in thinking:
                vlm_response.update({'Next Action': 'OPEN' + str([app])})
                return str(vlm_response)
    return str(vlm_response)

def process_image(
    image, max_pixels, min_pixels=None
) :
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    w,h = image.width,image.height
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image,w,h
def get_encoded_image(image_input):
    """将图片路径转换为 base64 编码的字符串。"""
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
    elif isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        image_input.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError("Input must be a file path or a PIL Image object.")
    return f"""data:image/png;base64,{encoded_image_text}"""


def process_action(step_info):
    next_action = step_info.get('Next Action')
    if 'CLICK' in next_action:
        coords = next_action.replace('CLICK', '')
        coords = coords.replace('[','')
        coords = coords.replace(']','')
        coords_ = coords.split(',')
        x_val = int(coords_[0].replace("'",""))
        y_val = int(coords_[1].replace("'",""))
        x_gt = x_val
        y_gt = y_val
        new_action = 'CLICK' + str([x_gt,y_gt])
        step_info.update({'Next Action': next_action})
    if 'LONG_PRESS' in next_action:
        coords = next_action.replace('LONG_PRESS', '')
        coords = coords.replace('[','')
        coords = coords.replace(']','')
        coords_ = coords.split(',')
        x_val = int(coords_[0].replace("'",""))
        y_val = int(coords_[1].replace("'",""))
        x_gt = x_val
        y_gt = y_val
        new_action = 'LONG_PRESS' + str([x_gt,y_gt])
        step_info.update({'Next Action': next_action})
    if 'SCROLL' in next_action:
        direction = next_action.replace('SCROLL','')
        direction = direction.replace('[','')
        direction = direction.replace(']','')
        if direction == 'UP':
            new_action = 'SCROLL' + str([540,1800,540,1200])
            step_info.update({'Next Action': next_action})
        if direction == 'DOWN':
            new_action = 'SCROLL' + str([540,600,540,1800])
            step_info.update({'Next Action': next_action})
        if direction == 'LEFT':
            new_action = 'SCROLL' +str([810,1200,270,1200])
            step_info.update({'Next Action': next_action})
        if direction == 'RIGHT':
            new_action = 'SCROLL' +str([270,1200,810,1200])
            step_info.update({'Next Action': next_action})
    return step_info 


TEST_DIR = 'data/ColorBench/data'

task_ids = os.listdir(TEST_DIR)

def get_benchmark_messages():
    benchmark_infos = []
    benchmark_messages = []
    print("Loading ColorBench TestData ...")
    for i in tqdm(range(len(task_ids))):
        task_id = task_ids[i]
        file_name = TEST_DIR + '/' + task_id + '/trajectory_v1_test.json'
        new_test_data = {}
        with open(file_name, 'r', encoding='utf-8') as f:
            testdata_ = json.load(f)
        testdata = testdata_[0]
        task = testdata.get('task')
        trajs = testdata.get('trajectories')
        history = ""
        for j in range(len(trajs)):
            current_traj = trajs[j]
            current_traj["task_id"] = task_id
            current_traj["task"] = task
            current_traj["step_idx"] = j
            if j == 0:
                action_des = current_traj['action_literal_des']
                user_content_list = [{"type": "text", "text": f'The user query: {task}\nTask progress (You have done the following operation on the current device): None. '}]
                #user_content_list = [{"type": "text", "text": "用户的任务:" + task}]
                current_screenshot,w,h = process_image(current_traj.get('image_path'),max_pixels=1024*1024)
                scalex,scaley=current_screenshot.size

                prompt_str = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {scalex}x{scaley}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform.", "enum": ["key", "click", "long_press", "swipe", "type", "clear_text", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, `action=answer` and `action=open`.", "type": "string"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>


# Output Format
Thought: ... (Your thinking process, explain your reasoning step-by-step )
Action: ... (Your action description, provide a brief description of the chosen action)
<answer>
[{{"name": <function-name>, "arguments": <args-json-object>}}]
</answer>"""
                prompt_str = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {scalex}x{scaley}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `clear_text`: Delete all characters (text or default Name) in the input field.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `answer`: Answer the user query.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "clear_text", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=key`, `action=type`, `action=answer` and `action=open`.", "type": "string"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>


# Output Format
Thought: ... (Your thinking process, explain your reasoning step-by-step )
Action: ... (Your action description, provide a brief description of the chosen action)
<answer>
[{{"name": <function-name>, "arguments": <args-json-object>}}]
</answer>"""

                user_content_list.append({"type": "image", "image": get_encoded_image(current_screenshot)})
                user_message = {"role": "user", "content": user_content_list}
                messages_for_processor = [{"role": "system", "content": prompt_str}, user_message]
                
                current_traj.update({'vlm_response': None,'w':w,'h':h,'scalex':scalex,'scaley':scaley})
                benchmark_messages.append(messages_for_processor)
                benchmark_infos.append(current_traj)

                history+=f'Step {j+1}: {action_des}; '
            else:
                #user_content_list = [{"type": "text", "text": "用户的任务:" + task}]
                user_content_list = [{"type": "text", "text": f'The user query: {task}\nTask progress (You have done the following operation on the current device): {history}. '}]
                
                current_screenshot,w,h = process_image(current_traj.get('image_path'),max_pixels=1024*1024)

                user_content_list.append({"type": "image", "image": get_encoded_image(current_screenshot)})
                user_message = {"role": "user", "content": user_content_list}
                messages_for_processor = [{"role": "system", "content": prompt_str}, user_message]

                current_traj.update({'vlm_response': None,'w':w,'h':h,'scalex':scalex,'scaley':scaley})
                benchmark_messages.append(messages_for_processor)
                benchmark_infos.append(current_traj)
                
                action_des = current_traj['action_literal_des']
                history+=f'Step {j+1}: {action_des}; '

    return benchmark_infos, benchmark_messages
