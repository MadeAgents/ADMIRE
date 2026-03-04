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

import os
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional

from openai_client import OpenClient
from PIL import Image, ImageDraw  # add PIL for rendering overlays

logger = logging.getLogger("step_helpfulness")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def default_system_prompt() -> str:
    return (
       '''You are a step helpfulness judge with precise evaluative capabilities: you can accurately determine whether a task has been completed, assess the relevance of actions to goals, and distinguish between meaningful progress and irrelevant actions. Your task is to evaluate a single-step trajectory of a GUI agent executing a task and determine whether this specific step is helpful for achieving the goal. 

        # What yor are provided
        You are given a mobile UI task goal, the current step's action text (including the model's thought process and output action), and the corresponding screenshots before and after the action. Red dots on the screenshots indicate click positions, while red arrows represent scroll operations and their directions. You are required to determine whether this step (action) helps progress toward completing the goal.

        If the task is completed at this step, its status is 'success', or the action is 'terminate', return 'yes' directly.
        If a discrepancy is found between the model's thought process and its final action, the action shall serve as the primary basis for judgment.
        Helpful steps include, but are not limited to: entering relevant information, advancing to the next stage, selecting target options, or saving progress.
        Unhelpful steps include: irrelevant taps, dismissing relevant dialogs, opening unrelated or wrong apps, idle, incomplete or wrong operations and so on.
        If there is uncertainty about whether it is helpful or the situation is ambiguous, Answer "no".
        If the action description or action text is empty or None, you should give 'no' directly.

        The following are some examples for your reference:

        Example for **helpful steps**:
        Task goal: Fill out an expense report.
        Action text: "Tap on the 'Add' button."
        Action description: "Click the 'Add' button at the top right corner of the page."
        Screenshot: A screenshot showing a button labeled 'Add' at the top right corner of the page  while a 'delete' button is on the left side of the page. The red dot is on the 'Add' button.
        Answer: "yes"


        Example for **unhelpful steps**:
        Task goal: Fill out an expense report.
        Action text: "Tap on the 'Add' button."
        Action description: "Click the 'Add' button at the top right corner of the page."
        Screenshot: A screenshot showing a button labeled 'Add' at the top right corner of the page while a 'delete' button is on the left side of the page. But the red dot is on the 'delete' button.
        Answer: "no"

        Example for **unhelpful steps**:
        Task goal: Fill out an expense report.
        Action text: ""
        Action description: ""
        Screenshot: A screenshot showing a button labeled 'Add' at the top right corner of the page while a 'delete' button is on the left side of the page.
        Answer: "no"
        
        The helpfulness of a step is not only determined by the action text and description, but also the screenshot. You should only give 'yes' for those steps that demonstrate clear and meaningful progress toward the goal. Be very strict and conservative - only give 'yes' when you are absolutely certain the step directly contributes to completing the task. If there is any doubt or uncertainty about whether a step is truly helpful, you must give 'no'.
        
        Your should **Answer strictly with 'yes' or 'no'** (lowercase) directly, without any explanation.
        '''
    )


def _parse_yes_no(response: str) -> str:
    r = (response or "").strip().lower()
    if "yes" in r:
        return "yes"
    if "no" in r:
        return "no"
    # fallback
    return "no"


class StepHelpfulnessJudge:
    def __init__(
        self,
        client: OpenClient,
        system_prompt: Optional[str] = None,
        max_pixels: Optional[int] = 1_500_000,
        temperature: float = 0.0,
        frequency_penalty: float = 1.0,
        resize_method: str = "qwen",
        api_delay: float = 0.5,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or default_system_prompt()
        self.max_pixels = max_pixels
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.resize_method = resize_method
        self.api_delay = api_delay
        # ref_task_path="milestones/task_name_to_data.json"
        # with open(ref_task_path, "r") as f:
        #     self.ref_task_dict = json.load(f)
        self.ref_task_dict = {}
    
    def update_ref_task_dict(self, replay_buffer,task):
        def _slice_task(replay_buffer, task):
            return replay_buffer[replay_buffer.non_tensor_batch["task"] == task] 

        def _get_goal_text(task_slice):
            goal_text = ""
            try:
                goals = task_slice.non_tensor_batch.get("goal", None)
                if goals is not None:
                    goals_list = goals.tolist() if hasattr(goals, "tolist") else goals
                    if isinstance(goals_list, list):
                        for g in goals_list:
                            if isinstance(g, str) and g.strip():
                                goal_text = g.strip()
                                break
                    elif isinstance(goals_list, str):
                        goal_text = goals_list
            except Exception:
                pass
            return goal_text

        def _get_action_descriptions(task_slice):
            try:
                action_desc_list = task_slice.non_tensor_batch.get("action_description", [])
                action_desc_list = action_desc_list.tolist() if hasattr(action_desc_list, "tolist") else action_desc_list
                if not isinstance(action_desc_list, list):
                    return []
                return action_desc_list
            except Exception:
                return []
        task_slice = _slice_task(replay_buffer,task)
        goal_text = _get_goal_text(task_slice)
        action_desc_list = _get_action_descriptions(task_slice)
        self.ref_task_dict[task] = {
            "task_goal": goal_text,
            "action_description": action_desc_list,
            "steps_num": len(action_desc_list),
        }


    # --- overlay helpers ---
    def _extract_click_point(self, args: Dict[str, Any]) -> Optional[tuple]:
        # supports {"x": int, "y": int} or {"coordinate": [x,y]}
        if isinstance(args, dict):
            if "x" in args and "y" in args:
                try:
                    return int(args["x"]), int(args["y"])
                except Exception:
                    pass
            coord = args.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    return int(coord[0]), int(coord[1])
                except Exception:
                    pass
        return None

    def _extract_swipe_points(self, args: Dict[str, Any]) -> Optional[tuple]:
        # supports start/end via coordinate/coordinate2 or explicit x1,y1,x2,y2
        if not isinstance(args, dict):
            return None
        start = None
        end = None
        # explicit x1,y1,x2,y2
        if all(k in args for k in ("x1", "y1", "x2", "y2")):
            try:
                start = (int(args["x1"]), int(args["y1"]))
                end = (int(args["x2"]), int(args["y2"]))
            except Exception:
                start = end = None
        # alternative x,y,x2,y2
        if start is None and all(k in args for k in ("x", "y", "x2", "y2")):
            try:
                start = (int(args["x"]), int(args["y"]))
                end = (int(args["x2"]), int(args["y2"]))
            except Exception:
                start = end = None
        if start is None:
            c1 = args.get("coordinate")
            c2 = args.get("coordinate2")
            if isinstance(c1, (list, tuple)) and len(c1) >= 2 and isinstance(c2, (list, tuple)) and len(c2) >= 2:
                try:
                    start = (int(c1[0]), int(c1[1]))
                    end = (int(c2[0]), int(c2[1]))
                except Exception:
                    start = end = None
        if start and end:
            return start, end
        return None

    def _annotate_image(self, image_path: str, action: Dict[str, Any], step_index: int) -> Optional[str]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            action_name = (action or {}).get("name", "")
            args_raw = (action or {}).get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {}

            # Only draw when we have something meaningful
            draw_click = action_name in {"click", "long_press"}
            draw_swipe = action_name in {"swipe", "drag","scroll"}

            if not (draw_click or draw_swipe):
                # Nothing to overlay
                return None

            im = Image.open(image_path).convert("RGBA")
            draw = ImageDraw.Draw(im)
            w, h = im.size

            if draw_click:
                pt = self._extract_click_point(args)
                if pt:
                    x, y = pt
                    # enlarge click marker
                    r = 28 if action_name == "click" else 34
                    # outer ring (white) for contrast
                    draw.ellipse((x - r - 4, y - r - 4, x + r + 4, y + r + 4), outline=(255, 255, 255, 255), width=8)
                    # inner filled red circle
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0, 150), outline=(255, 0, 0, 255), width=3)

            if draw_swipe:
                pts = self._extract_swipe_points(args)
                direction = str(args.get("direction", "")).lower()
                
                # Fallback: handle scroll with only (x,y) + direction
                if not pts and direction:
                    start_pt = self._extract_click_point(args)
                    if start_pt:
                        # gesture length
                        L = max(80, int(min(w, h) * 0.4))
                        # For up/down, we draw them in canonical positions (bottom-up, top-down)
                        # For left/right, we draw them centered vertically.
                        if direction in ("up", "u"):
                            x1, y1 = w // 2, int(h * 0.8)
                            x2, y2 = x1, y1 - L
                        elif direction in ("down", "d"):
                            x1, y1 = w // 2, int(h * 0.2)
                            x2, y2 = x1, y1 + L
                        elif direction in ("left", "l"):
                            x1, y1 = int(w * 0.8), h // 2
                            x2, y2 = x1 - L, y1
                        elif direction in ("right", "r"):
                            x1, y1 = int(w * 0.2), h // 2
                            x2, y2 = x1 + L, y1
                        else: # No direction, cannot draw
                            x1 = y1 = x2 = y2 = -1 # Invalid coordinates

                        if x1 != -1: # check if direction was valid
                            # clamp
                            x1 = max(0, min(w - 1, x1))
                            y1 = max(0, min(h - 1, y1))
                            x2 = max(0, min(w - 1, x2))
                            y2 = max(0, min(h - 1, y2))
                            pts = ((x1, y1), (x2, y2))

                # if no points, try to use direction
                if not pts:
                    direction = str(action.get("direction", "")).lower()
                    if direction:
                        w, h = im.size
                        L = int(min(w, h) * 0.4)  # Arrow length as 40% of smaller dimension
                        x1, y1, x2, y2 = -1, -1, -1, -1 # init with invalid

                        if direction in ("up", "u"):
                            # Arrow at the top, pointing up
                            x1, y1 = w // 2, int(h * 0.2) + L
                            x2, y2 = w // 2, int(h * 0.2)
                        elif direction in ("down", "d"):
                            # Arrow at the bottom, pointing down
                            x1, y1 = w // 2, int(h * 0.8) - L
                            x2, y2 = w // 2, int(h * 0.8)
                        elif direction in ("left", "l"):
                            # Arrow in the middle, pointing left
                            x1, y1 = int(w * 0.5) + L // 2, h // 2
                            x2, y2 = int(w * 0.5) - L // 2, h // 2
                        elif direction in ("right", "r"):
                            # Arrow in the middle, pointing right
                            x1, y1 = int(w * 0.5) - L // 2, h // 2
                            x2, y2 = int(w * 0.5) + L // 2, h // 2

                        if x1 != -1:  # check if direction was valid
                            # clamp coordinates to be within image bounds
                            x1 = max(0, min(w - 1, x1))
                            y1 = max(0, min(h - 1, y1))
                            x2 = max(0, min(w - 1, x2))
                            y2 = max(0, min(h - 1, y2))
                            pts = ((x1, y1), (x2, y2))

                if pts:
                    (x1, y1), (x2, y2) = pts
                    # draw main stroke thicker with white underlay for contrast
                    stroke_w = 16
                    under_w = stroke_w + 6
                    red = (255, 0, 0, 255)
                    white = (255, 255, 255, 230)
                    draw.line((x1, y1, x2, y2), fill=white, width=under_w)
                    draw.line((x1, y1, x2, y2), fill=red, width=stroke_w)
                    # arrow head aligned with line angle (with underlay)
                    try:
                        import math
                        ah_len = 34
                        ah_angle = math.radians(28)
                        theta = math.atan2(y2 - y1, x2 - x1)
                        left_theta = theta + math.pi - ah_angle
                        right_theta = theta + math.pi + ah_angle
                        lx = x2 + ah_len * math.cos(left_theta)
                        ly = y2 + ah_len * math.sin(left_theta)
                        rx = x2 + ah_len * math.cos(right_theta)
                        ry = y2 + ah_len * math.sin(right_theta)
                        # underlay
                        draw.line((x2, y2, lx, ly), fill=white, width=under_w)
                        draw.line((x2, y2, rx, ry), fill=white, width=under_w)
                        # overlay
                        draw.line((x2, y2, lx, ly), fill=red, width=stroke_w)
                        draw.line((x2, y2, rx, ry), fill=red, width=stroke_w)
                    except Exception:
                        # fallback simple head
                        ah = 26
                        draw.line((x2, y2, x2 - ah, y2 - ah), fill=white, width=under_w)
                        draw.line((x2, y2, x2 - ah, y2 + ah), fill=white, width=under_w)
                        draw.line((x2, y2, x2 - ah, y2 - ah), fill=red, width=stroke_w)
                        draw.line((x2, y2, x2 - ah, y2 + ah), fill=red, width=stroke_w)

            # save next to source under a dedicated folder
            out_dir = os.path.join(os.path.dirname(image_path), "_annotated_clicks")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_dir, f"{base}-s{step_index}.png")
            im.convert("RGB").save(out_path, format="PNG")
            return out_path
        except Exception as e:
            logger.warning(f"Failed to annotate image {image_path}: {e}")
            return None

    # --- core judging ---
    def judge_step(self, goal: str, action_text: str, image_path: Optional[str], action_description: str = "") -> Dict[str, Any]:
        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Step Action Text:\n{action_text}\n\n"
            f"Step Action Description:\n{action_description}\n\n"
            f"Question: Does this step help progress toward the goal? Answer 'yes' or 'no' directly."
        )

        image_paths: List[str] = []
        if image_path and os.path.exists(image_path):
            image_paths = [image_path]
        else:
            if image_path:
                logger.warning(f"Image not found: {image_path}. Proceeding without image.")

        try:
            resp = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                image_paths=image_paths,
                max_pixels=self.max_pixels,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                stop=["\n"],
                resize_method=self.resize_method,
            )
            label = _parse_yes_no(resp)
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            resp = ""
            label = "no"

        return {
            "label": label,
            "raw_response": resp,
        }

    def judge_step_with_reference(self, goal: str, action_text: str, image_path: Optional[str], action_description: str = "",task:str="") -> Dict[str, Any]:
        reference_traj=self.ref_task_dict[task]["action_description"]
        # use 'step i' to connect the reference trajectory and the current step
        reference_traj_str=""
        for i in range(len(reference_traj)):
            reference_traj_str+=f"Step {i+1}: {reference_traj[i]}\n"
        
        user_prompt = (
            f"Before you judge, I will give you a reference trajectory for the task '{task}', which is a totally successful trajectory and might have slight differences(e.g. different file name, different date, different content) with the current task. You should judge the helpfulness of the step based on the reference trajectory and the current step. \n\n"
            f"The reference trajectory is as follows: \n{reference_traj_str}\n\n"
            f"You should not rely on the reference trajectory too much, but it can be a reference for you to judge the helpfulness of the current step. \n\n"
            f"Now, I will give you the current task and step, you should judge the helpfulness of the step based on the reference trajectory and the current step. \n\n"
            f"Goal: {goal}\n\n"
            f"Step Action Text:\n{action_text}\n\n"
            f"Step Action Description:\n{action_description}\n\n"
            f"Question: Does this step help progress toward the goal? Answer 'yes' or 'no' directly."
        )

        image_paths: List[str] = []
        if image_path and os.path.exists(image_path):
            image_paths = [image_path]
        else:
            if image_path:
                logger.warning(f"Image not found: {image_path}. Proceeding without image.")

        try:
            resp = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                image_paths=image_paths,
                max_pixels=self.max_pixels,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                stop=["\n"],
                resize_method=self.resize_method,
            )
            label = _parse_yes_no(resp)
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            resp = ""
            label = "no"

        return {
            "label": label,
            "raw_response": resp,
        }

    def judge_step_with_neighbor(
        self,
        goal: str,
        action_text: str,
        image_path: Optional[str],
        action_description: str = "",
        task: str = "",
        prev_step: Optional[Dict[str, Any]] = None,
        next_step: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reference_traj: List[str] = []
        if task and task in self.ref_task_dict:
            reference_traj = self.ref_task_dict[task].get("action_description", [])

        reference_traj_str = ""
        for i, desc in enumerate(reference_traj):
            reference_traj_str += f"Step {i + 1}: {desc}\n"

        if reference_traj_str:
            reference_section = (
                f"The reference trajectory is as follows: \n{reference_traj_str}\n"
            )
        else:
            reference_section = "Reference trajectory is not available for this task.\n\n"

        ordered_images: List[str] = []
        image_counter = 1

        def build_step_section(title: str, step_info: Optional[Dict[str, Any]]) -> str:
            nonlocal image_counter
            if not isinstance(step_info, dict):
                return ""
            step_action_text = step_info.get("action_text", "")
            step_action_desc = step_info.get("action_description", "")
            step_image_path = step_info.get("image_path")
            screenshot_text = "Screenshot: Not provided."
            if step_image_path and os.path.exists(step_image_path):
                screenshot_text = f"Screenshot: Image {image_counter}."
                ordered_images.append(step_image_path)
                image_counter += 1
            return (
                f"{title}:\n"
                f"Action Text: {step_action_text}\n"
                f"Action Description: {step_action_desc}\n"
                f"{screenshot_text}"
            )

        prev_section = build_step_section("Neighbor Reference - Previous Step", prev_step)
        next_section = build_step_section("Neighbor Reference - Next Step", next_step)
        prev_section_text = (
            prev_section
            if prev_section
            else "Neighbor Reference - Previous Step:\nNo previous step provided."
        )
        next_section_text = (
            next_section
            if next_section
            else "Neighbor Reference - Next Step:\nNo next step provided."
        )

        target_section = build_step_section(
            "Target Step (evaluate this step)",
            {
                "action_text": action_text,
                "action_description": action_description,
                "image_path": image_path,
            },
        )
        if not target_section:
            target_section = (
                "Target Step (evaluate this step):\n"
                "Action Text: \n"
                "Action Description: \n"
                "Screenshot: Not provided."
            )

        user_prompt = (
            f"Before you judge, I will give you a reference trajectory for the task '{task}', which is a totally successful trajectory and might have slight differences(e.g. different file name, different date, different content) with the current task. \n\n"
            f"{reference_section}"
            f"Avoid over-reliance on this reference trajectory, but it can serve as a reference when evaluating the helpfulness of the current step. \n\n"
            f"Below are the previous and next steps. You should assess the helpfulness of the current step based on these steps. \n\n"
            f"{prev_section_text}\n\n"
            f"{next_section_text}\n\n"
            f"The provided previous and next steps are solely for helping you more accurately evaluate the helpfulness of the current step. Their own helpfulness should not be considered when judging the current step. \n\n"
            f"Now, I will give you the current task and step, you should judge the helpfulness of the step based on the reference trajectory, previous and next steps. \n\n"
            f"Goal: {goal}\n\n"
            f"Step Action Text:\n{action_text}\n\n"
            f"Step Action Description:\n{action_description}\n\n"
            f"Question: Does this step help progress toward the goal? Answer 'yes' or 'no' directly."
        )

        try:
            resp = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                image_paths=ordered_images,
                max_pixels=self.max_pixels,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                stop=["\n"],
                resize_method=self.resize_method,
            )
            label = _parse_yes_no(resp)
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            resp = ""
            label = "no"

        return {
            "label": label,
            "raw_response": resp,
        }

    def judge_trajectory(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        task = traj.get("task", "").strip()
        goal = traj.get("goal", "").strip()
        action_texts: List[str] = traj.get("action_text", [])
        action_descs: List[str] = traj.get("action_description", [])
        actions: List[Dict[str, Any]] = traj.get("action", [])
        image_paths: List[str] = traj.get("image_path", [])

        n_steps = min(len(action_texts), len(actions))
        results: List[Dict[str, Any]] = []

        for i in range(n_steps):
            action_text = action_texts[i]
            action_desc = action_descs[i] if i < len(action_descs) else ""
            img_path = image_paths[i] if i < len(image_paths) else (image_paths[-1] if image_paths else None)

            # prepare annotated image with click/swipe overlay if applicable
            annotated_path = self._annotate_image(img_path, actions[i] if i < len(actions) else {}, i) if img_path else None
            effective_img_path = annotated_path or img_path
            judged = self.judge_step(
                goal=goal,
                action_text=action_text,
                image_path=effective_img_path,
                action_description=action_desc,
            )
            if task not in self.ref_task_dict:
                judged = self.judge_step(
                    goal=goal,
                    action_text=action_text,
                    image_path=effective_img_path,
                    action_description=action_desc,
                )
            else:
                # prev_step_payload = None
                # next_step_payload = None

                # if i - 1 >= 0:
                #     prev_step_payload = {
                #         "action_text": action_texts[i - 1],
                #         "action_description": action_descs[i - 1] if i - 1 < len(action_descs) else "",
                #         "image_path": (
                #             image_paths[i - 1]
                #             if i - 1 < len(image_paths)
                #             else (image_paths[-1] if image_paths else None)
                #         ),
                #     }

                # if i + 1 < len(action_texts):
                #     next_step_payload = {
                #         "action_text": action_texts[i + 1],
                #         "action_description": action_descs[i + 1] if i + 1 < len(action_descs) else "",
                #         "image_path": (
                #             image_paths[i + 1]
                #             if i + 1 < len(image_paths)
                #             else (image_paths[-1] if image_paths else None)
                #         ),
                #     }

                # judged = self.judge_step_with_neighbor(
                #     goal=goal,
                #     action_text=action_text,
                #     image_path=effective_img_path,
                #     action_description=action_desc,
                #     task=task,
                #     prev_step=prev_step_payload,
                #     next_step=next_step_payload,
                # )
                judged = self.judge_step_with_reference(
                    goal=goal,
                    action_text=action_text,
                    image_path=effective_img_path,
                    action_description=action_desc,
                    task=task,
                )
            # Collect rich per-step info for downstream analysis
            step_entry: Dict[str, Any] = {
                "step_index": i,
                "goal": goal,
                "action_text": action_text,
                "action_description": action_desc,
                "action": actions[i] if i < len(actions) else {},
                "image_path": img_path or "",
                "annotated_image_path": annotated_path or "",
                "label": judged["label"],
                "raw_response": judged["raw_response"],
            }
            results.append(step_entry)

            if self.api_delay and i < n_steps - 1:
                time.sleep(self.api_delay)

        return {
            "goal": goal,
            "num_steps": n_steps,
            "results": results,
        }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_client(address: str, model_name: str, api_key: str = "EMPTY") -> OpenClient:
    return OpenClient(address=address, model_name=model_name, api_key=api_key)


def infer_output_path(input_path: str, out_dir: Optional[str] = None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_name = base + "_step_helpfulness.json"
    if out_dir:
        return os.path.join(out_dir, out_name)
    return os.path.join(os.path.dirname(input_path), out_name)


def main():
    parser = argparse.ArgumentParser(description="Judge per-step helpfulness for a trajectory JSON")
    parser.add_argument("--json_path", default="saves/trajectory.json",help="Path to the trajectory JSON (one complete trajectory)")
    parser.add_argument("--address", default="YOUR_API_BASE_URL", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--api_key", default="YOUR_API_KEY", help="API key")
    parser.add_argument("--out_dir", default="saves/step_judge", help="Directory to save results")
    parser.add_argument("--max_pixels", type=int, default=1_500_000, help="Max pixels for screenshot resize")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--freq_penalty", type=float, default=1.0)
    parser.add_argument("--resize_method", default="qwen")
    parser.add_argument("--api_delay", type=float, default=0.5)

    args = parser.parse_args()

    # client = build_client(address=args.address, model_name=args.model, api_key=args.api_key)
    client = build_client(address="YOUR_API_BASE_URL", model_name="qwen2.5-vl-72b-instruct")
    judge = StepHelpfulnessJudge(
        client=client,
        system_prompt=default_system_prompt(),
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        frequency_penalty=args.freq_penalty,
        resize_method=args.resize_method,
        api_delay=args.api_delay,
    )

    traj = load_json(args.json_path)
    summary = judge.judge_trajectory(traj)

    out_path = infer_output_path(args.json_path, args.out_dir)
    save_json(summary, out_path)
    logger.info(f"Saved per-step helpfulness to {out_path}")

    # Also print a concise table summary to stdout
    for r in summary["results"]:
        print(f"step={r['step_index']:02d} label={r['label']}")


if __name__ == "__main__":
    main()
