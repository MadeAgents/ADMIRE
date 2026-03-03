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
import re
from copy import deepcopy

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    if predicted_str == ground_truth_str:
        return 1

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def extract_arguments(predict_str: str) -> dict or str:
    """
    适配新格式：从「thought###reasoning###action」中提取动作与参数
    返回：字典（含action及对应参数）或"no action"（提取失败）
    """
    try:
        # 步骤1：提取action部分（匹配"action: "后的内容）
        action_match = re.search(r"action:\s*(.*?)(?:\s*#|$)", predict_str, re.DOTALL)
        if not action_match:
            return "no action2"
        action_str = action_match.group(1).strip()  # 如"CLICK[300,620]"、"TYPE[深圳]"
        
        # 步骤2：按动作类型解析参数（覆盖所有新动作）
        # 2.1 CLICK[x,y]：提取坐标
        if 'CLICK' in action_str:
            # pass#
            action_str=action_str.replace('[[','[').replace(']]',']').replace('，',',').replace('CLICK: CLICK','CLICK').replace('CLICK"','CLICK')
            if '[' not in action_str:
                if 'CLICK: '  in action_str:
                    action_str = action_str.replace('CLICK: ','CLICK[')
                else:
                    
                    action_str = action_str.replace('CLICK:','CLICK[')
            if ']' not in action_str:
                action_str+=']'
        click_match = re.match(r"CLICK\s*\[*(\d+),\s*(\d+)\]", action_str)
        if click_match:
            x, y = click_match.groups()
            return {"action": "click", "coordinate": [int(x), int(y)]}
        
        # 2.2 TYPE[text]：提取输入文本
        type_match = re.match(r"TYPE\[([^\[\]]+)\]", action_str)
        if type_match:
            text = type_match.group(1).strip()
            return {"action": "type", "text": text}
        
        # 2.3 SWIPE[x1,y1,x2,y2]：提取起始/结束坐标
        swipe_match = re.match(r"SWIPE\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", action_str)
        if swipe_match:
            x1, y1, x2, y2 = swipe_match.groups()
            return {
                "action": "swipe",
                "coordinate": [int(x1), int(y1)],  # 起始点
                "coordinate2": [int(x2), int(y2)]  # 结束点
            }
        
        # 2.4 open[APP_NAME]：提取应用名
        open_match = re.match(r"open\[([^\[\]]+)\]", action_str)
        if open_match:
            app_name = open_match.group(1).strip()
            return {"action": "open", "text": app_name}
        
        # 2.5 LONG_PRESS[x,y]：提取长按坐标
        long_press_match = re.match(r"LONG_PRESS\[(\d+),\s*(\d+)\]", action_str)
        if long_press_match:
            x, y = long_press_match.groups()
            return {"action": "long_press", "coordinate": [int(x), int(y)]}
        
        # 2.6 SYSTEM_BUTTON[category]：提取系统按钮类型
        sys_btn_match = re.match(r"SYSTEM_BUTTON\[([^\[\]]+)\]", action_str)
        if sys_btn_match:
            category = sys_btn_match.group(1).strip()
            # 校验按钮类型合法性（新格式限定Back/Home/Enter）
            if category in ["Back", "Home", "Enter"]:
                return {"action": "system_button", "button": category}
            else:
                return "no action"
        
        # 2.7 无参数动作：WAIT/COMPLETE/TASK_IMPOSSIBLE
        if "WAIT" in action_str :
            return {"action": "wait"}
        elif "COMPLETE" in action_str:
            return {
                "action": "terminate",
                "status": "success"
            }
            # return {"action": "COMPLETE"}  # 映射原terminate-success
        elif "TASK_IMPOSSIBLE" in action_str:
            return {
                "action": "terminate",
                "status": "failure"
            }
            # return {"action": "TASK_IMPOSSIBLE"}  # 映射原terminate-failure
        
        # 未匹配任何动作格式
        return "no action"
    
    except Exception as e:
        # 捕获正则解析、类型转换等异常
        return "no action"

def benchmark_evaluate(benchmark_infos, benchmark_responses, output_path, global_train_step_idx):

    total_steps = len(benchmark_infos)
    type_cnt = 0
    match_cnt = 0  
    early = 0 
    click_false = 0
    wrong_click_data = []

    # 新增：存储匹配失败的案例
    failure_cases = []

    # 初始化各类动作的错误计数器
    action_error_counts = {
        'OPEN': 0,
        'CLICK': 0,
        'WAIT': 0,
        'INPUT_TEXT': 0,
        'COMPLETE': 0,
        'SCROLL': 0,
        'SYSTEM_BUTTON': 0,
        'LONG_PRESS': 0,
        'OTHER': 0  # 其他未分类的动作
    }
        
    for b_idx, _step in enumerate(benchmark_infos):
        step = deepcopy(_step)
        vlm_response = benchmark_responses[b_idx]
        step["vlm_response"] = vlm_response

        step_idx = step.get("step_idx", -2)
        task_id = step.get("task_id", "")
        task = step.get("task", "")
        model_name = "Qwen2.5"
        gt_action = step.get('gt_action', '')
        step_info = step.get('step_info', '无步骤信息')
        image_path = step["image_path"]
            
        try:
            # action = extract_arguments(vlm_response)
            action_match = re.search(r'<answer>(.*?)</answer>', vlm_response, flags=re.DOTALL)
            
            if not action_match:
                print('no action',vlm_response)
                continue
            action_s = action_match.group(1).strip()
            action = json.loads(action_s)
            action = action[0]['arguments']
            name = action['action']
            
            # name = action['action']
            if name in ['click','long_press']:
                w, h, scalex, scaley = step.get('w'), step.get('h'), step.get('scalex'), step.get('scaley')
                action['coordinate'][0] = action['coordinate'][0] / scalex * w
                action['coordinate'][1] = action['coordinate'][1] / scaley * h


            predicted_action_detail = f"{name}: {action}"  # 更详细的预测动作信息
        except:
            print(vlm_response)
            continue
        
        # 处理各类动作的匹配与错误计数
        action_type = None
        is_success = False
        
        if 'OPEN' in gt_action:
            action_type = 'OPEN'
            
            
            if 'click' == name:
                type_cnt += 1

                match_cnt += 1

                is_success = True
            elif 'open' == name:
                type_cnt += 1
                match_cnt += 1
                app_name = gt_action.replace('OPEN', '').replace('[','').replace(']','').strip('\'').strip('"')
                try:
                    model_app_name = action['text'].strip(' ')
                except:
                    print(vlm_response)
                # is_success = True
                if app_name == model_app_name:
                    # match_cnt += 1
                    is_success = True
                else:
                    # print(app_name)
                    # print(model_app_name)
                    # print('------------------------------------------------\n')
                    action_error_counts[action_type] += 1
            else:
                action_error_counts[action_type] += 1
                
        elif 'CLICK' in gt_action:
            action_type = 'CLICK'
            if 'terminate' == name:
                early += 1
                action_error_counts[action_type] += 1
            elif 'click' == name:
                type_cnt += 1
                bbox = step.get('ui_bbox')
                image_path = step.get('image_path')
                
                if not bbox:
                    match_cnt += 1
                    is_success = True
                    continue 
                    
                x_min = bbox.get('x_min')
                x_max = bbox.get('x_max')
                y_min = bbox.get('y_min')
                y_max = bbox.get('y_max')
                
                coords_ = action['coordinate']
                if len(coords_) != 2:
                    action_error_counts[action_type] += 1
                else:
                    x_val = coords_[0]
                    y_val = coords_[1]
                    # w, h, scalex, scaley = step.get('w'), step.get('h'), step.get('scalex'), step.get('scaley')
                    
                    # if w and h and scalex and scaley:  # 确保所有缩放参数都存在
                    #     x_val = x_val / scalex * w
                    #     y_val = y_val / scaley * h
                        
                    if x_val >= x_min and x_val <= x_max and y_val >= y_min and y_val <= y_max:
                        match_cnt += 1
                        is_success = True
                    else:
                        click_false += 1
                        action_error_counts[action_type] += 1
                    
            else:
                action_error_counts[action_type] += 1
                
        elif 'WAIT' in gt_action:
            action_type = 'WAIT'
            
            if 'wait' == name:
                type_cnt += 1
                match_cnt += 1
                is_success = True
            else:
                action_error_counts[action_type] += 1
                
        elif 'INPUT_TEXT' in gt_action:
            action_type = 'INPUT_TEXT'
            
            if 'type' != name:
                wrong_click_data.append({
                    "task":task,
                    'task_id': task_id,
                    'vlm_response': vlm_response,
                    'step_info': step_info,
                    'gt_action': gt_action,
                    'image_path': step.get('image_path'),
                    'bbox': step.get('ui_bbox')
                })
                action_error_counts[action_type] += 1
            else:
                type_cnt += 1
                match_cnt += 1
                gt_text = gt_action.replace('INPUT_TEXT', '').replace('[', '').replace(']', '').strip('\'').strip('"')
                model_text = action['text'].strip('\'').strip('"')
                # is_success = True
                if gt_text == model_text:
        
                    # match_cnt += 1
                    is_success = True
                else:
                    # print(gt_text)
                    # print(model_text)
                    # print('-------------------------------------------------------\n')
                    action_error_counts[action_type] += 1
                    
        elif 'COMPLETE' in gt_action:
            action_type = 'COMPLETE'
            
            if 'terminate' == name:
                type_cnt += 1
                match_cnt += 1
                is_success = True
            else:
                action_error_counts[action_type] += 1
                
        elif 'SCROLL' in gt_action:
            action_type = 'SCROLL'
            if 'swipe' == name:
                type_cnt += 1
                gt_dir = gt_action.replace('SCROLL', '').replace('[', '').replace(']', '').strip('\'').strip('"')
                
                if 1:
                    #gt_dir = gt_dir.replace('[', '').replace(']', '').strip('\'').strip('"')
                    coords_ = action['coordinate'] + action['coordinate2']
                    x1, y1, x2, y2 = coords_
                    det_x = abs(x1 - x2)
                    det_y = abs(y1 - y2)
                    if 'Qwen3' in model_name or 'qwen3' in model_name:
                        if det_x < det_y and y1 > y2:
                            model_dir ='DOWN' #'UP'
                        elif det_x < det_y and y1 < y2:
                            model_dir = 'UP'#'DOWN'
                        elif det_x > det_y and x1 > x2:
                            model_dir = 'LEFT'
                        elif det_x > det_y and x1 < x2:
                            model_dir = 'RIGHT'
                        else:
                            model_dir = ''
                    else:
                        if det_x < det_y and y1 < y2:
                            model_dir ='DOWN' #'UP'
                        elif det_x < det_y and y1 > y2:
                            model_dir = 'UP'#'DOWN'
                        elif det_x > det_y and x1 > x2:
                            model_dir = 'LEFT'
                        elif det_x > det_y and x1 < x2:
                            model_dir = 'RIGHT'
                        else:
                            model_dir = ''
                        
                    if model_dir == gt_dir:
                        match_cnt += 1
                        is_success = True
                    else:
                        action_error_counts[action_type] += 1
                        print(model_dir , gt_dir)
                    
            else:
                action_error_counts[action_type] += 1
                
        elif 'SYSTEM_BUTTON' in gt_action:
            action_type = 'SYSTEM_BUTTON'
            
            if 'system_button' == name:
                type_cnt += 1
                gt_cate = gt_action.replace('SYSTEM', '').strip('\'').strip('"')
                model_cate = action['button']
                if gt_cate == model_cate:
                    match_cnt += 1
                    is_success = True
                else:
                    action_error_counts[action_type] += 1
            else:
                action_error_counts[action_type] += 1
                
        elif 'LONG_PRESS' in gt_action:
            action_type = 'LONG_PRESS'
            
            if 'long_press' == name:
                type_cnt += 1
                bbox = step.get('ui_bbox')
                if not bbox:
                    match_cnt += 1
                    is_success = True
                    continue 
                    
                x_min = bbox.get('x_min')
                x_max = bbox.get('x_max')
                y_min = bbox.get('y_min')
                y_max = bbox.get('y_max')
                
                coords_ = action['coordinate']
                if len(coords_) != 2:
                    action_error_counts[action_type] += 1
                else:
                    x_val = coords_[0]
                    y_val = coords_[1]
                    # w, h, scalex, scaley = step.get('w'), step.get('h'), step.get('scalex'), step.get('scaley')
                    
                    # if w and h and scalex and scaley:  # 确保所有缩放参数都存在
                    #     x_val = x_val / scalex * w
                    #     y_val = y_val / scaley * h
                        
                    if x_val >= x_min and x_val <= x_max and y_val >= y_min and y_val <= y_max:
                        match_cnt += 1
                        is_success = True
                    else:
                        action_error_counts[action_type] += 1
                    
            else:
                action_error_counts[action_type] += 1
                
        else:
            # 未识别的动作类型
            action_type = 'OTHER'
            action_error_counts[action_type] += 1
        
        # 如果匹配失败，记录失败案例
        if not is_success and action_type:
            failure_cases.append({
                "task":task,
                'task_id': task_id,
                'step': step_idx + 1,
                'step_info': step_info,
                'gt_action': gt_action,
                'predicted_action': predicted_action_detail,
                'error_type': action_type
            })

    # 保存错误数据
    with open(f"{output_path}/step{global_train_step_idx}-wrong-click-data.json", 'w', encoding='utf-8') as g:
        json.dump(wrong_click_data, g, indent=4, ensure_ascii=False)

    # 新增：保存失败案例到txt文件
    with open(f"{output_path}/step{global_train_step_idx}-failure-cases.txt", 'w', encoding='utf-8') as f:
        f.write("模型预测失败案例记录\n")
        f.write("=" * 100 + "\n")
        for idx, case in enumerate(failure_cases, 1):
            f.write(f"案例 {idx}:\n")
            f.write(f"任务: {case['task']}\n")
            # f.write(f"步骤: {case['step']}\n")
            # f.write(f"步骤信息: {case['step_info']}\n")
            f.write(f"真实动作: {case['gt_action']}\n")
            f.write(f"预测动作: {case['predicted_action']}\n")
            # f.write(f"错误类型: {case['error_type']}\n")
            f.write("-" * 100 + "\n")

    # 输出统计结果
    print(f"提前终止次数: {early}")
    print(f"总步骤数: {total_steps}, 匹配类型数: {type_cnt}, 匹配成功数: {match_cnt}")
    type_acc = round(type_cnt/total_steps if total_steps > 0 else 0, 4)
    print(f"类型匹配率: {type_acc}")
    match_acc = round(match_cnt/total_steps if total_steps > 0 else 0, 4)
    print(f"动作成功率: {match_acc}")
    print(f"点击错误数: {click_false}")

    # 输出各类动作的错误统计
    print("\n各类动作错误统计:")
    for action_type, count in action_error_counts.items():
        print(f"{action_type}: {count} 次错误")

    print(f"\n失败案例已记录到: {output_path}/step{global_train_step_idx}-wrong-click-data.json")
    return {
        "early": early,
        "type_cnt / total_steps": type_acc,
        "match_cnt / total_steps": match_acc,
        "click_false": click_false
    } | action_error_counts
