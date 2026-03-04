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
import numpy as np
from verl.protocol import DataProto


class MilestoneGenerator:
    def __init__(self, replay_buffer, milestone_dict, max_images=5, base_url=None, model_name=None, api_key=None):
        self.replay_buffer = replay_buffer
        self.milestone_dict = milestone_dict
        self.max_images = max_images
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self.model_name = model_name or os.getenv("MILESTONE_MODEL", "gpt-4o")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.graph_dict = {}
        
        # Initialize OpenAI client
        try:
            from openai_client import OpenClient
            self.client = OpenClient(address=self.base_url, model_name=self.model_name, api_key=self.api_key)
        except Exception as e:
            print(f"[Warn] Failed to initialize OpenClient: {e}")
            self.client = None

    @staticmethod
    def _dedupe_preserve_order(items):
        seen = set()
        result = []
        for it in items:
            if it and it not in seen:
                seen.add(it)
                result.append(it)
        return result

    @staticmethod
    def _parse_json_list(text):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        if text.strip().startswith("```") and text.strip().endswith("```"):
            lines = text.strip().splitlines()
            body = "\n".join(lines[1:-1])
            try:
                parsed = json.loads(body)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []
    
    @staticmethod
    def _parse_nested_list(text):
        """
        Parse nested list (list of lists) for milestone links.
        
        Args:
            text: Model response text, should be in format like "[[1,2],[2,3],[3],[]]"
        
        Returns:
            Parsed nested list, or None if parsing fails
        """
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, list):
                        return None
                return parsed
        except Exception:
            pass

        if text.strip().startswith("```") and text.strip().endswith("```"):
            lines = text.strip().splitlines()
            body = "\n".join(lines[1:-1])
            try:
                parsed = json.loads(body)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, list):
                            return None
                    return parsed
            except Exception:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, list):
                            return None
                    return parsed
            except Exception:
                pass
        
        return None

    @staticmethod
    def _extract_image_paths(values, max_images=5):
        paths = []

        def push(p):
            if isinstance(p, str) and p and os.path.exists(p):
                paths.append(p)

        if values is None:
            return []

        try:
            values_list = values.tolist() if hasattr(values, "tolist") else values
        except Exception:
            values_list = values

        if isinstance(values_list, list):
            for v in values_list:
                if isinstance(v, (list, np.ndarray)):
                    try:
                        sub_list = v.tolist() if hasattr(v, "tolist") else v
                    except Exception:
                        sub_list = v
                    for s in sub_list:
                        push(s)
                else:
                    push(v)
        else:
            push(values_list)

        unique = []
        seen = set()
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[-max_images:]

    def _slice_task(self, task):
        return self.replay_buffer[self.replay_buffer.non_tensor_batch["task"] == task]

    def _get_goal_text(self, task_slice):
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

    def _get_action_descriptions(self, task_slice):
        try:
            action_desc_list = task_slice.non_tensor_batch.get("action_description", [])
            action_desc_list = action_desc_list.tolist() if hasattr(action_desc_list, "tolist") else action_desc_list
            if not isinstance(action_desc_list, list):
                return []
            return action_desc_list
        except Exception:
            return []

    def _build_milestone_prompt(self, task, goal_text, action_desc_list):
        SYSTEM_PROMPT = (
            '''
                You are an expert in Android system operations. You will be given a successfully completed task trajectory, including:

                * the task name and overall goal,
                * a sequence of action descriptions representing user interactions, and
                * corresponding reference screenshots.

                Your task is to identify only the **essential steps** required to accomplish the goal. Focus exclusively on actions that are absolutely necessary, and **ignore redundant or non-essential steps**.

                Your objective is to distill these essential actions into a **concise list of milestones** representing the key sub-tasks needed to complete the goal. These milestones:

                * should represent **logically distinct stages** of progress toward the goal,
                * must be **clear**,
                * should be **short, non-overlapping, and non-redundant**, and
                * must include **only necessary steps**.
                * Keep important action-related keywords unchanged.
                

                When task-specific details appear (such as names, dates, events, entries, files, or paths), replace them with generic placeholders like `<Name>`, `<Date>`, `<Event>`, `<Entry>`, `<File>`, or `<Path>`.

                Your final output must be a **strict JSON array** containing only these milestone strings.
                Do **not** include any numbering, explanations, commentary, or other non-JSON content.

                Focus on summarizing the **core actions** into a clean, abstract milestone list — generalizable and free of specific details.
                For example: "Locate the <Entry> on the page", "Input <Name> as the note name", "Navigate to the <Path> on the page".
  
            '''
        )

        action_desc_list = [s.strip() for s in action_desc_list if isinstance(s, str) and s.strip()]
        action_desc_list = self._dedupe_preserve_order(action_desc_list)

        user_prompt = (
            f"User Instruction: {goal_text if goal_text else task}\n\n"
            "Example action descriptions (in order; may be redundant; for reference only):\n" +
            ("\n".join(f"- {s}" for s in action_desc_list) if action_desc_list else "- (none)") +
            "\n\nOutput the essential milestone list (<= 10 steps) as a strict JSON array."
        )
        return SYSTEM_PROMPT, user_prompt

    def _build_update_prompt(self, task, goal_text, previous_milestone, action_desc_list):
        SYSTEM_PROMPT = (
            '''
                You are an expert in refining mobile UI task milestones. You will be provided with the following information:

                * the task name and overall goal,
                * an existing milestone list,
                * a set of action descriptions from a newly observed successful trajectory, and
                * corresponding reference screenshots.

                Your role is to analyze the new trajectory, filtering out any redundant or meaningless interactions and focusing only on actions that are **essential** for achieving the task's objective.

                Your primary responsibility is to determine whether the **existing milestone list** can be improved based on the new information.
                By default, assume the current milestones are already sufficient. They should be revised **only** if the new trajectory clearly reveals **more minimal, precise, or effective** actions that better represent the essential steps.
                If the existing milestone list already provides a clear, general, and minimal set of essential steps, return it **unchanged**.

                If an update is warranted, output the improved milestones as a **strict JSON array**.
                Each milestone must be:

                * a short, **executable step** starting with an action verb,
                * **non-redundant** and **clearly phrased**, and
                * generalized by replacing task-specific details (e.g., names, dates, events, entries, paths) with placeholders such as `<Name>`, `<Date>`, `<Event>`, `<Entry>`, or `<Path>`.

                Do **not** include any numbering, commentary, or text outside the JSON structure.
                Your output must contain **only** the finalized milestones in valid JSON format.
            '''
        )

        action_desc_list = [s.strip() for s in action_desc_list if isinstance(s, str) and s.strip()]
        action_desc_list = self._dedupe_preserve_order(action_desc_list)
        prev_str = json.dumps(previous_milestone, ensure_ascii=False, indent=2)

        user_prompt = (
            f"Task: {task}\n"
            f"User Instruction: {goal_text if goal_text else task}\n\n"
            f"Existing milestone (JSON):\n{prev_str}\n\n"
            "Example action descriptions from new trajectory (ordered; may be redundant):\n" +
            ("\n".join(f"- {s}" for s in action_desc_list) if action_desc_list else "- (none)") +
            "\n\nDecide whether to keep the existing milestone or refine it. Output ONLY the final milestone as a strict JSON array."
        )
        return SYSTEM_PROMPT, user_prompt

    def _build_link_prompt(self, milestone:list[str]):
        SYSTEM_PROMPT = (
            '''
            You are an expert in analyzing task workflows and identifying potential transition paths in milestone sequences. You will be provided with a list of milestones that represent the steps to complete a task.

            Your task is to determine all possible direct transition paths from each milestone to later milestones. 

            Important rules:
            1. Adjacent milestones (milestone i to milestone i+1) can ALWAYS transition directly. This is guaranteed.
            2. You need to analyze whether non-adjacent milestones (milestone i to milestone i+2, i+3, etc.) can also have direct transitions, meaning you can skip intermediate steps.
            3. A direct skip connection exists if completing milestone i naturally allows you to proceed directly to milestone j (where j > i+1), without necessarily completing all intermediate milestones.
            4. Consider the semantic meaning and dependencies of each milestone when making your decision.

            Output format:
            - Return a JSON array where each element corresponds to a milestone (in order, starting from index 0).
            - Each element should be a list of ALL indices that can be directly reached from the current milestone, including the immediate next one (i+1) AND any valid skip connections (i+2, i+3, etc.).
            - Use 0-based indexing.
            - If a milestone can only reach the next adjacent one, include only [i+1].
            - The last milestone should have an empty array [] as it cannot transition anywhere.
            
            Example:
            If there are 4 milestones (0, 1, 2, 3), and milestone 0 can skip to milestone 2 but not 3, milestone 1 can skip to milestone 3, the output would be:
            [[1, 2], [2, 3], [3], []]
            This means:
            - Milestone 0 can go to 1 (adjacent) and 2 (skip)
            - Milestone 1 can go to 2 (adjacent) and 3 (skip)
            - Milestone 2 can only go to 3 (adjacent)
            - Milestone 3 cannot go anywhere (end)
            
            Output ONLY the JSON array, with no additional text or explanation.
            '''
        )

        milestone_str = "\n".join(f"{i}. {m}" for i, m in enumerate(milestone))
        
        user_prompt = (
            "Task milestones:\n" +
            milestone_str +
            "\n\nAnalyze all possible transitions from each milestone. "
            "Remember: adjacent transitions (i to i+1) always exist. "
            "Determine if any skip transitions (i to i+2, i+3, etc.) are also possible. "
            "Output a JSON array showing ALL reachable milestones for each milestone."
        )
        
        return SYSTEM_PROMPT, user_prompt
        

    def _call_model(self, system_prompt, user_prompt, image_paths):
        if self.client is None:
            print(f"[Error] OpenClient not initialized")
            return []
        
        try:
            response = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                image_paths=image_paths,
                max_pixels=1024 * 1024,
                max_tokens=512,
                temperature=0.0,
                frequency_penalty=0,
                stop=[]
            )
            return self._parse_json_list(response)
        except Exception as e:
            print(f"[Error] MLLM call failed: {e}")
            return []
    
    def _call_model_for_links(self, system_prompt, user_prompt):
        """
        Call model to generate milestone links.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
        
        Returns:
            Parsed nested list, or None if failed
        """
        if self.client is None:
            print(f"[Error] OpenClient not initialized")
            return None
        
        try:
            response = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                image_paths=[],
                max_pixels=1024 * 1024,
                max_tokens=512,
                temperature=0.0,
                frequency_penalty=0,
                stop=[]
            )
            return self._parse_nested_list(response)
        except Exception as e:
            print(f"[Error] Model call for links failed: {e}")
            return None

    def generate_from_scratch(self, task):
        task_slice = self._slice_task(task)
        goal_text = self._get_goal_text(task_slice)
        action_desc_list = self._get_action_descriptions(task_slice)
        image_values = task_slice.non_tensor_batch.get("image_path", None)
        image_paths = self._extract_image_paths(image_values, max_images=self.max_images)
        system_prompt, user_prompt = self._build_milestone_prompt(task, goal_text, action_desc_list)
        milestone_list = self._call_model(system_prompt, user_prompt, image_paths)
        if not milestone_list:
            print("[Warn] Model response could not be parsed as a JSON array; returning empty list.")
            return milestone_list, goal_text, image_paths, False
        return milestone_list, goal_text, image_paths, True

    def generate_from_previous(self, task):
        previous = self.milestone_dict.get(task)
        if not previous:
            print(f"[Warn] No previous milestone found for task: {task}")
            return self.generate_from_scratch(task)

        task_slice = self._slice_task(task)
        goal_text = self._get_goal_text(task_slice)
        action_desc_list = self._get_action_descriptions(task_slice)
        image_values = task_slice.non_tensor_batch.get("image_path", None)
        image_paths = self._extract_image_paths(image_values, max_images=self.max_images)
        system_prompt, user_prompt = self._build_update_prompt(task, goal_text, previous, action_desc_list)
        milestone_list = self._call_model(system_prompt, user_prompt, image_paths)
        if not milestone_list:
            print("[Warn] Model response could not be parsed as a JSON array; falling back to previous milestone.")
            return previous, goal_text, image_paths, False
        return milestone_list, goal_text, image_paths, True

    def generate_milestone_links(self, milestone):
        """
        Generate milestone connection relationships.
        
        Args:
            milestone: Milestone list, e.g., ["Step 1", "Step 2", "Step 3"]
        
        Returns:
            Connection relationship list (nested list), e.g., [[1, 2], [2, 3], [3], []]
            If parsing fails, return default sequential connections (each connects only to next adjacent node)
        """
        if not milestone or len(milestone) == 0:
            print("[Warn] Empty milestone list provided.")
            return []
        
        if len(milestone) == 1:
            return [[]]
        
        system_prompt, user_prompt = self._build_link_prompt(milestone)
        
        links = self._call_model_for_links(system_prompt, user_prompt)
        
        if links is None or not isinstance(links, list) or len(links) != len(milestone):
            print(f"[Warn] Model response format invalid. Expected list of length {len(milestone)}, got: {links}")
            print("[Info] Falling back to default sequential connections.")
            default_links = [[i+1] for i in range(len(milestone)-1)] + [[]]
            return default_links
        
        for i, link in enumerate(links):
            if not isinstance(link, list):
                print(f"[Warn] Invalid format at index {i}: {link}. Expected a list.")
                print("[Info] Falling back to default sequential connections.")
                default_links = [[i+1] for i in range(len(milestone)-1)] + [[]]
                return default_links
        
        print("[Info] Successfully generated milestone links.")
        return links

    def save_milestone_list(self, task_name, goal_text, milestone_list, image_paths=None):
        out_path = "milestones/generated_milestones.json"
        payload = {
            "task": task_name,
            "goal": goal_text,
            "milestone_list": milestone_list,
            "images": image_paths or []
        }
        try:
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            data[task_name] = payload
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[Info] Milestone list saved to: {out_path}")
        except Exception as e:
            print(f"[Error] Failed to save milestone list: {e}")

    
    def init_graph_from_scratch(self, task):
        milestone_list, goal_text, image_paths, success = self.generate_from_scratch(task)
        
        if not success or not milestone_list:
            return None
        
        milestones_with_virtual = ["<BEGIN>"] + milestone_list + ["<END>"]
        n = len(milestones_with_virtual)
        
        graph = {
            "task": task,
            "milestones": milestones_with_virtual,
            "adjacency": {str(i): [i+1] if i < n-1 else [] for i in range(n)}
        }
        
        self.graph_dict[task] = graph
        print(f"[Info] Initialized single-chain graph with {len(milestone_list)} milestones (+ BEGIN/END)")
        return graph
    
    def update_graph(self, task):
        if task not in self.graph_dict:
            print(f"[Warn] No graph found for {task}, initializing from scratch")
            return self.init_graph_from_scratch(task)
        
        graph = self.graph_dict[task]
        
        new_milestones, goal_text, image_paths, success = self.generate_from_scratch(task)
        
        if not success or not new_milestones:
            print("[Warn] Failed to generate milestones from new trajectory")
            return graph
        
        print(f"[Info] Generated {len(new_milestones)} milestones from new trajectory")
        
        result = self._classify_alignment_type(graph, new_milestones)
        result["new_milestones"] = new_milestones
        print(f"[Debug] Classification result: {result}")
        
        if result["type"] == "full_match":
            print(f"[Info] Full match - trajectory matches existing path")
        
        elif result["type"] == "text_optimize":
            self._handle_text_optimize(graph, new_milestones, result)
        
        elif result["type"] == "new_path":
            self._handle_new_path(graph, new_milestones, result)
        
        elif result["type"] == "shortcut":
            self._handle_shortcut(graph, new_milestones, result)
        
        return graph
    
    
    def _build_graph_representation(self, graph):
        lines = ["Existing Milestone Graph:", ""]
        
        for node_id in range(len(graph["milestones"])):
            milestone = graph["milestones"][node_id]
            next_nodes = graph["adjacency"].get(str(node_id), [])
            
            next_str = f" -> {next_nodes}" if next_nodes else " (end)"
            lines.append(f"Node {node_id}: {milestone}{next_str}")
        
        return "\n".join(lines)
    
    def _classify_alignment_type(self, graph, new_milestones):
        SYSTEM_PROMPT = '''
            You are an expert in analyzing task milestone relationships.
            
            Given an existing milestone graph and a new milestone sequence, you need to:
            1. Align new milestones to existing nodes (by semantic meaning)
            2. Classify the relationship type
            
            Important - Two Numbering Systems:
            1. **Milestone Graph Node IDs**: 
               - Nodes in the existing graph are identified by Node IDs (e.g., Node 0, Node 1, Node 5)
               - Node IDs do NOT necessarily represent sequential order (it's a graph, not a linear sequence)
               - Node 0 is always "<BEGIN>", last node is always "<END>"
               
            2. **New Milestone Indices**:
               - New milestones from the trajectory are indexed sequentially (0, 1, 2, ...)
               - These indices ALWAYS represent sequential order in the trajectory
            
            Graph Format:
            - Each node: "Node <ID>: <milestone text> -> [list]"
            - Arrow "->" shows which Node IDs can be reached next
            - Example: "Node 0: Open app -> [1, 2]" means can proceed to Node 1 or Node 2
            - "(end)" means terminal node
            
            Alignment Rules:
            - Map new milestone indices to graph Node IDs based on semantic meaning
            - Two milestones match if they describe the SAME action/goal (even with different wording)
            - Order matters: follow sequential order of new milestones
            - Some new milestones may not match any node (unmatched)
            - Do NOT match new milestones to <BEGIN> or <END> nodes
            
            Four relationship types:
            
            1. full_match: New trajectory follows existing path completely
               - All new milestones matched to existing nodes
               - No unmatched new milestones
            
            2. new_path: New trajectory contains alternative steps
               - Some new milestones have no match in the graph
               - Represents a different way to accomplish part of the task
            
            3. shortcut: New trajectory reveals a shorter path in existing graph
               - All new milestones match existing nodes
               - But the matched nodes can be connected more directly
               - Example: If graph has A→B→C→D, and new trajectory shows A→D is possible,
                 this is a shortcut (not because B,C are on different paths, but because we can skip them)
            
            4. text_optimize: All matched but milestone text needs optimization
               - All new milestones matched to existing nodes
               - But new trajectory has better/clearer wording
               - Should update milestone descriptions
            
            Output:
            - alignment_map: a dictionary mapping new milestone indices (keys) to graph Node IDs (values)
            - type: the relationship type
            - reason: A brief explanation for the selection of the type.


            Example output with strict JSON format:
            {
                "alignment_map": {
                    "0": 1,    // new milestone index 0 -> graph Node ID 1
                    "1": 2     // new milestone index 1 -> graph Node ID 2
                },
                "type": "new_path",
                "reason": "The new trajectory contains alternative steps, so it is a new path."
            }
        '''
        
        graph_repr = self._build_graph_representation(graph)
        new_milestones_str = "\n".join(f"{i}. {m}" for i, m in enumerate(new_milestones))
        
        user_prompt = f"""
            {graph_repr}
            
            New Milestone Sequence:
            {new_milestones_str}
            
            Analyze alignment and determine relationship type.
            Output ONLY the JSON result.
        """
        
        if self.client is None:
            print(f"[Error] OpenClient not initialized")
            return {"type": "full_match", "alignment_map": {}}
        
        try:
            response = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                image_paths=[],
                max_pixels=1024 * 1024,
                max_tokens=512,
                temperature=0.0,
                frequency_penalty=0,
                stop=[]
            )
            result = self._parse_json_dict(response)
            
            if not result or "type" not in result or "alignment_map" not in result:
                print(f"[Error] Failed to parse result, skipping update")
                return {"type": "full_match", "alignment_map": {}}
            
            alignment_map = result.get("alignment_map", {})
            result_type = result.get("type", "full_match")
            
            print(f"[Info] Type: {result_type} - {result.get('reason', '')}")
            
            matched_new_indices = set(int(k) for k in alignment_map.keys())
            unmatched_new = [i for i in range(len(new_milestones)) if i not in matched_new_indices]
            
            if result_type == "full_match":
                return {
                    "type": "full_match",
                    "alignment_map": alignment_map
                }
            elif result_type == "new_path":
                return {
                    "type": "new_path",
                    "alignment_map": alignment_map,
                    "unmatched_new": unmatched_new
                }
            elif result_type == "shortcut":
                return {
                    "type": "shortcut",
                    "alignment_map": alignment_map
                }
            else:  # text_optimize
                return {
                    "type": "text_optimize",
                    "alignment_map": alignment_map
                }
                
        except Exception as e:
            print(f"[Error] Classification failed: {e}")
            return {"type": "full_match", "alignment_map": {}}
    
    
    def _handle_text_optimize(self, graph, new_milestones, alignment):
        alignment_map = alignment.get("alignment_map", {})
        
        if self.client is None:
            print(f"[Warn] OpenClient not initialized, skipping optimization check")
            return
        
        for new_idx_str, old_idx in alignment_map.items():
            new_idx = int(new_idx_str)
            old_text = graph["milestones"][old_idx]
            
            if old_text in ["<BEGIN>", "<END>"]:
                continue
            
            new_text = new_milestones[new_idx]
            
            should_optimize, optimized_text = self._should_optimize_text(old_text, new_text)
            
            if should_optimize:
                graph["milestones"][old_idx] = optimized_text
                print(f"[Info] Optimized Node {old_idx}: '{old_text}' -> '{optimized_text}'")
    
    def _should_optimize_text(self, old_text, new_text):
        SYSTEM_PROMPT = '''
            You are an expert in text optimization for task milestones.
            
            Given two milestone descriptions that describe the SAME action:
            - Old text: existing milestone in the graph
            - New text: milestone from a new trajectory
            
            Determine if the new text is BETTER and should replace the old text.
            
            A text is better if it is:
            1. More concise while keeping key information
            2. Clearer and easier to understand
            3. More precise in describing the action
            4. Uses simpler vocabulary
            
            If the new text is better, return true and the optimized text.
            
            Output strict JSON:
            {
                "should_optimize": true | false,
                "optimized_text": "the text to use (either new text or a refined version)",
            }
        '''
        
        user_prompt = f"""
            Old text: "{old_text}"
            New text: "{new_text}"
            
            Should we optimize? If yes, provide the optimized text.
            Output ONLY the JSON result.
        """
        
        if self.client is None:
            return False, old_text
        
        try:
            response = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                image_paths=[],
                max_pixels=1024 * 1024,
                max_tokens=256,
                temperature=0.0,
                frequency_penalty=0,
                stop=[]
            )
            result = self._parse_json_dict(response)
            
            if result and "should_optimize" in result:
                should_opt = result["should_optimize"]
                opt_text = result.get("optimized_text", new_text)
                print(f"[Debug] Optimization decision: {should_opt} - {result.get('reason', '')}")
                return should_opt, opt_text
            else:
                return False, old_text
                
        except Exception as e:
            print(f"[Error] Text optimization failed: {e}")
            return False, old_text
    
    def _handle_new_path(self, graph, new_milestones, alignment):
        unmatched_indices = alignment.get("unmatched_new", [])
        alignment_map = alignment.get("alignment_map", {})
        
        if not unmatched_indices:
            return
        
        if self.client is None:
            print(f"[Warn] OpenClient not initialized, cannot add new path")
            return
        
        graph_repr = self._build_graph_representation(graph)
        
        new_milestones_str = "\n".join(f"{i}. {m}" for i, m in enumerate(new_milestones))
        
        alignment_info = []
        for new_idx_str, old_node_id in sorted(alignment_map.items(), key=lambda x: int(x[0])):
            new_idx = int(new_idx_str)
            alignment_info.append(f"New[{new_idx}] -> Node {old_node_id}")
        
        unmatched_info = [f"New[{i}]: {new_milestones[i]}" for i in unmatched_indices]
        
        SYSTEM_PROMPT = '''
            You are an expert in graph structure analysis.
            
            Given:
            - Existing milestone graph (includes virtual <BEGIN> and <END> nodes)
            - New milestone sequence with alignment to existing nodes
            - Unmatched new milestones that MAY form an alternative path
            
            Important - Two Numbering Systems:
            1. **Graph Node IDs**: Used for existing milestone graph (e.g., Node 1, Node 5)
               - Node IDs do NOT represent sequential order (it's a graph structure)
               - Node 0 is always <BEGIN>, last node is always <END>
               
            2. **New Milestone Indices**: Used for new trajectory milestones (e.g., New[0], New[1])
               - Indices ALWAYS represent sequential order in the trajectory (0, 1, 2, ...)
            
            Task: Determine if there is a valid alternative path and how to insert it.
            
            Note:
            - Alternative paths should diverge/converge at real milestone nodes, not virtual nodes
            - diverge_node and converge_node should be graph Node IDs
            - insert_milestone_indices should be new milestone indices
            
            Analysis steps:
            1. Check if unmatched milestones form a CONTINUOUS alternative branch
            2. The alternative path must:
               - Start after some matched milestone (diverge point)
               - End before another matched milestone (converge point)
               - Represent a different way to achieve the same sub-goal
            
            3. Not all unmatched milestones should be inserted:
               - Only include those that form the alternative branch
               - Exclude unmatched milestones at the start/end that don't form a branch
               - Exclude noise or irrelevant steps
            
            Output strict JSON:
            {
                "has_alternative_path": true/false,
                "diverge_node": <graph Node ID or null>,
                "converge_node": <graph Node ID or null>,
                "insert_milestone_indices": [<list of new milestone indices>]
            }

            Example output:
            {
                "has_alternative_path": true,
                "diverge_node": 1,          // graph Node ID where path diverges
                "converge_node": 3,         // graph Node ID where path converges
                "insert_milestone_indices": [4, 5]  // new milestone indices to insert
            }
            
            If has_alternative_path is false, set diverge_node, converge_node to null and insert_milestone_indices to empty list.
        '''
        
        user_prompt = f"""
            {graph_repr}
            
            New Milestone Sequence:
            {new_milestones_str}
            
            Alignment (matched pairs):
            {chr(10).join(alignment_info) if alignment_info else "(none)"}
            
            Unmatched new milestones:
            {chr(10).join(unmatched_info)}
            
            Analyze if there is a valid alternative path, and determine:
            - diverge_node and converge_node (from existing graph nodes)
            - which unmatched milestones should be inserted
            
            Output ONLY the JSON result.
        """
        
        try:
            response = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                image_paths=[],
                max_pixels=1024 * 1024,
                max_tokens=512,
                temperature=0.0,
                frequency_penalty=0,
                stop=[]
            )
            result = self._parse_json_dict(response)
            
            if not result or "has_alternative_path" not in result:
                print(f"[Error] Failed to parse alternative path decision")
                return
            
            if not result["has_alternative_path"]:
                print(f"[Info] No valid alternative path found")
                print(f"[Info] Reason: {result.get('reason', 'N/A')}")
                return
            
            diverge_node = result.get("diverge_node")
            converge_node = result.get("converge_node")
            insert_indices = result.get("insert_milestone_indices", [])
            
            if diverge_node is None or converge_node is None or not insert_indices:
                print(f"[Warn] Invalid alternative path specification")
                return
            
            if diverge_node >= len(graph["milestones"]) or converge_node >= len(graph["milestones"]):
                print(f"[Error] Invalid diverge/converge nodes: {diverge_node}, {converge_node}")
                return
            
            if not all(0 <= idx < len(new_milestones) for idx in insert_indices):
                print(f"[Error] Invalid insert milestone indices: {insert_indices}")
                return
            
            print(f"[Info] Valid alternative path found: Node {diverge_node} -> ... -> Node {converge_node}")
            print(f"[Info] Reason: {result.get('reason', 'N/A')}")
            
            new_path_milestones = [new_milestones[i] for i in insert_indices]
            self._add_alternative_path(graph, diverge_node, converge_node, new_path_milestones)
            print(f"[Info] Added {len(new_path_milestones)} new milestone(s) as alternative path")
            
        except Exception as e:
            print(f"[Error] Failed to handle new path: {e}")
    
    def _handle_shortcut(self, graph, new_milestones, alignment):
        alignment_map = alignment.get("alignment_map", {})
        
        if len(alignment_map) < 2:
            return
        
        if self.client is None:
            print(f"[Warn] OpenClient not initialized, cannot handle shortcut")
            return
        
        matched_sequence = []
        for new_idx_str, old_node_id in sorted(alignment_map.items(), key=lambda x: int(x[0])):
            new_idx = int(new_idx_str)
            matched_sequence.append({
                "new_idx": new_idx,
                "new_text": new_milestones[new_idx],
                "old_node": old_node_id,
                "old_text": graph['milestones'][old_node_id]
            })
        
        graph_repr = self._build_graph_representation(graph)
        
        matched_sequence_str = "\n".join([
            f"New[{m['new_idx']}] -> Node {m['old_node']}: \"{m['old_text']}\""
            for m in matched_sequence
        ])
        
        SYSTEM_PROMPT = '''
            You are an expert in graph optimization.
            
            Given:
            - An existing milestone graph with its adjacency structure
            - A new trajectory that matches some nodes in the graph
            - The matched node sequence represents a valid path to complete the task
            
            Important - Two Numbering Systems:
            1. **Graph Node IDs**: Used for existing milestone graph (e.g., Node 1, Node 5)
               - Node IDs do NOT represent sequential order (it's a graph structure)
               - These are what appear in the graph adjacency lists
               
            2. **New Milestone Indices**: Used for new trajectory milestones (e.g., New[0], New[1])
               - Indices ALWAYS represent sequential order in the trajectory (0, 1, 2, ...)
               
            The matched sequence shows: New[i] -> Node X, meaning new milestone i matches graph Node X
            
            Task: Identify shortcut opportunities.
            
            A SHORTCUT exists when:
            - Two graph nodes in the matched sequence (e.g., Node A and Node C) are NOT directly connected in the graph
            - But the new trajectory shows they CAN be connected directly (skipping intermediate steps)
            - Example: Graph has A→B→C, new trajectory shows A can go directly to C
            
            Important:
            - Analyze the matched node sequence and the graph adjacency
            - For consecutive nodes in matched sequence, check if they are already directly connected
            - If NOT directly connected, but reachable through a path, consider adding a shortcut edge
            - Do NOT add shortcuts involving <BEGIN> or <END> nodes
            - Do NOT add shortcuts that already exist
            
            Also consider if milestone text should be optimized based on the new trajectory.
            
            Output:
            - shortcuts: a list of shortcuts to add (using graph Node IDs)
            - optimizations: a list of optimizations to apply (using graph Node IDs)

            Output strict JSON:
            {
                "shortcuts": [
                    {
                        "from_node": <graph Node ID>, 
                        "to_node": <graph Node ID>, 
                        "reason": "explanation"
                    }
                ],
                "optimizations": [
                    {
                        "node_id": <graph Node ID>,
                        "new_text": "optimized text",
                        "reason": "explanation"
                    }
                ]
            }
            
            Note: ALL node references (from_node, to_node, node_id) should be graph Node IDs, NOT new milestone indices.
            
            Return empty arrays if no shortcuts or optimizations are needed.
        '''
        
        user_prompt = f"""
            {graph_repr}
            
            New trajectory matched sequence:
            {matched_sequence_str}
            
            Analyze if any shortcut edges should be added between nodes in the matched sequence.
            Check the graph adjacency to see which nodes are already directly connected.
            Output ONLY the JSON result.
        """
        
        try:
            response = self.client.get_completion(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                image_paths=[],
                max_pixels=1024 * 1024,
                max_tokens=512,
                temperature=0.0,
                frequency_penalty=0,
                stop=[]
            )
            result = self._parse_json_dict(response)
            
            if not result:
                print(f"[Warn] Failed to parse shortcut result")
                return
            
            shortcuts = result.get("shortcuts", [])
            optimizations = result.get("optimizations", [])
            
            for shortcut in shortcuts:
                from_node = shortcut.get("from_node")
                to_node = shortcut.get("to_node")
                reason = shortcut.get("reason", "")
                
                if from_node is None or to_node is None:
                    continue
                    
                if from_node >= len(graph["milestones"]) or to_node >= len(graph["milestones"]):
                    print(f"[Warn] Invalid shortcut nodes: {from_node} -> {to_node}")
                    continue
                
                if graph["milestones"][from_node] in ["<BEGIN>", "<END>"]:
                    continue
                if graph["milestones"][to_node] in ["<BEGIN>", "<END>"]:
                    continue
                
                if str(from_node) not in graph["adjacency"]:
                    graph["adjacency"][str(from_node)] = []
                
                if to_node not in graph["adjacency"][str(from_node)]:
                    graph["adjacency"][str(from_node)].append(to_node)
                    print(f"[Info] Added shortcut: Node {from_node} -> Node {to_node}")
                    print(f"       Reason: {reason}")
                else:
                    print(f"[Info] Shortcut already exists: Node {from_node} -> Node {to_node}")
            
            for opt in optimizations:
                node_id = opt.get("node_id")
                new_text = opt.get("new_text")
                reason = opt.get("reason", "")
                
                if node_id is None or not new_text:
                    continue
                    
                if node_id >= len(graph["milestones"]):
                    continue
                
                old_text = graph["milestones"][node_id]
                if old_text in ["<BEGIN>", "<END>"]:
                    continue
                    
                graph["milestones"][node_id] = new_text
                print(f"[Info] Optimized Node {node_id}: '{old_text}' -> '{new_text}'")
                print(f"       Reason: {reason}")
            
            if shortcuts:
                print(f"[Info] Total shortcuts added: {len(shortcuts)}")
            if optimizations:
                print(f"[Info] Total optimizations applied: {len(optimizations)}")
                        
        except Exception as e:
            print(f"[Error] Failed to handle shortcut: {e}")
    
    
    def _parse_json_dict(self, text):
        try:
            return json.loads(text)
        except:
            pass
        
        if "```" in text:
            start = text.find("```")
            end = text.rfind("```")
            if start != -1 and end != -1:
                content = text[start+3:end]
                if content.startswith("json"):
                    content = content[4:]
                try:
                    return json.loads(content.strip())
                except:
                    pass
        
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        
        return None
    
    def _add_alternative_path(self, graph, diverge_node, converge_node, new_milestones):
        if not new_milestones:
            return
        
        n = len(graph["milestones"])
        new_node_ids = []
        
        for i, milestone in enumerate(new_milestones):
            new_id = n + i
            graph["milestones"].append(milestone)
            new_node_ids.append(new_id)
        
        if str(diverge_node) not in graph["adjacency"]:
            graph["adjacency"][str(diverge_node)] = []
        if new_node_ids[0] not in graph["adjacency"][str(diverge_node)]:
            graph["adjacency"][str(diverge_node)].append(new_node_ids[0])
        
        for i in range(len(new_node_ids) - 1):
            graph["adjacency"][str(new_node_ids[i])] = [new_node_ids[i + 1]]
        
        graph["adjacency"][str(new_node_ids[-1])] = [converge_node]
        
        print(f"[Info] Added alternative path: Node {diverge_node} -> [{', '.join(map(str, new_node_ids))}] -> Node {converge_node}")
    
    
    def visualize_graph(self, task):
        if task not in self.graph_dict:
            print(f"[Warn] No graph found for task: {task}")
            return
        
        graph = self.graph_dict[task]
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        
        print("Graph Structure:")
        for node_id in range(len(graph["milestones"])):
            milestone = graph["milestones"][node_id]
            next_nodes = graph["adjacency"].get(str(node_id), [])
            next_str = f"-> {next_nodes}" if next_nodes else "(end)"
            
            if milestone in ["<BEGIN>", "<END>"]:
                print(f"  [{node_id}] {milestone} (virtual) {next_str}")
            else:
                print(f"  [{node_id}] {milestone} {next_str}")
        
        print(f"\n{'='*60}\n")
    

    

# if __name__ == "__main__":
#     cur_task = os.getenv("CUR_TASK", "FilesMoveFile")
#     generator = MilestoneGenerator(
#         replay_buffer=replay_buffer,
#         milestone_dict=milestone_dict,
#         max_images=5,
#         base_url=os.getenv("OPENAI_BASE_URL","YOUR_API_BASE_URL"),
#         model_name=os.getenv("MILESTONE_MODEL", "qwen2.5-vl-72b-instruct"),
#         api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
#     )
#     milestone_list, goal_text, image_paths = generator.generate_from_previous(cur_task)
#     print("[Result] Milestone list:")
#     print(milestone_list)


# "ExpenseAddSingle": [
# "Open 'Pro Expense' application",
# "Enter <Name> in Name field",
# "Enter <Amount> in Amount field",
# "Select <Category> as category",
# "Enter <Note> in Note field",
# "Save the expense"
# ],


# if __name__ == "__main__":
#     milestone = [
#     "Open Tasks app",
#     "View tasks due next week",
#     "Count tasks due next week"
#     ]
#     generator = MilestoneGenerator(
#         replay_buffer=None,
#         milestone_dict=None,
#         max_images=5,
#         base_url=os.getenv("OPENAI_BASE_URL","YOUR_API_BASE_URL"),
#         model_name=os.getenv("MILESTONE_MODEL", "gpt-4o"),
#         api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
#     )
#     links = generator.generate_milestone_links(milestone)
#     print(links)