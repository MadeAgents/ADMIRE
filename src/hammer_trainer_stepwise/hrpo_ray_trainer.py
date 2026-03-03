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

import logging
import uuid
import math
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import traceback
import time
from datetime import datetime
import random
from typing import Optional, Dict, List, Any, Tuple
import os
import re

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.utils.dataset.rl_dataset import collate_fn

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    _timer,
)

# Import the judge and client builder
from judge_step_helpfulness import StepHelpfulnessJudge, build_client
from replayTomilestone import MilestoneGenerator
from openai_client import OpenClient
import numpy as np
import ray
import torch
import uuid
import math
import json

from hammer_trainer_stepwise.env import EnvWorker
# from hammer_trainer.metric_utils import compute_data_metrics
from hammer_trainer.utils.dataset.rl_dataset import MessagesDataset2 as MessagesDataset
from hammer_trainer.utils.dataset.rl_dataset import collate_fn_dummy
from hammer_trainer_stepwise.utils import get_batch_human_helps, get_batch_reflection
from hammer_trainer_stepwise.utils import pad_dataproto_to_constant_size
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import glob
from skimage.metrics import structural_similarity as ssim
from hammer_trainer_stepwise.colorbench_inference import get_benchmark_messages
from hammer_trainer_stepwise.colorbench_evaluate import benchmark_evaluate
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None
INIT_TASK_TIMEOUT = 10 * 60
logger = logging.getLogger(__name__)

class BertEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

class RayHRPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        # collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""
        super(RayHRPOTrainer, self).__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn_dummy,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        self.env_workers = []
        self.model_variant = getattr(self.config.env, "model_variant", "hammer")
        self.human_helps = None
        # self.benchmark_infos, self.benchmark_messages = get_benchmark_messages()
        # os.makedirs(f"{self.config.trainer.validation_data_dir}/static", exist_ok=True)
        
        # path to store per-step [confidence_reward, accuracy_reward] pairs
        # prefer config.data.np_path, fallback to ./outputs
        try:
            root_dir = self.config.data.get("np_path", None)
        except Exception:
            root_dir = None
        if not root_dir:
            root_dir = "outputs"
        try:
            os.makedirs(root_dir, exist_ok=True)
        except Exception:
            pass
        self.conf_acc_npy_path = os.path.join(root_dir, "thought_confidence_accuracy_rewards.npy")
        self.conf_ent_acc_npy_path = os.path.join(root_dir, "thought_confidence_entropy_accuracy_rewards.npy")
        self.act_conf_acc_npy_path = os.path.join(root_dir, "action_confidence_accuracy_rewards.npy")
        self.conf_count_npy_path = os.path.join(root_dir, "confidence_count.npy")
        self.kl_acc_help_npy_path = os.path.join(root_dir, "kl_uniform_accuracy_helpfulness.npy")

        # Initialize the StepHelpfulnessJudge

        judge_config = self.config.step_judge
        judge_client = build_client(
            address=judge_config.address,
            model_name=judge_config.model,
            api_key=judge_config.api_key,
        )
        self.step_judge = StepHelpfulnessJudge(
            client=judge_client,
            max_pixels=judge_config.max_pixels,
            temperature=judge_config.temperature,
            resize_method=judge_config.resize_method,
            api_delay=judge_config.api_delay,
        )
        logger.info(f"StepHelpfulnessJudge initialized with model {judge_config.model}")

        # Initialize MilestoneJudge for model-based milestone matching
        if hasattr(self.config, 'milestone_judge'):
            milestone_judge_config = self.config.milestone_judge
            milestone_judge_client = OpenClient(
                address=milestone_judge_config.get('address', judge_config.address),
                model_name=milestone_judge_config.get('model', judge_config.model),
                api_key=milestone_judge_config.get('api_key', judge_config.api_key),
            )
            self.milestone_judge = MilestoneJudge(
                client=milestone_judge_client,
                temperature=milestone_judge_config.get('temperature', 0.0),
                frequency_penalty=milestone_judge_config.get('frequency_penalty', 1.0),
            )
            logger.info(f"MilestoneJudge initialized with model {milestone_judge_config.get('model', judge_config.model)}")
        else:
            # Fallback: use same config as step_judge
            self.milestone_judge = MilestoneJudge(
                client=judge_client,
                temperature=0.0,
                frequency_penalty=1.0,
            )
            logger.info(f"MilestoneJudge initialized with model {judge_config.model} (using step_judge config)")
        
        
        self.bert_embedder = BertEmbedder()
        # cache for milestone images per task
        self._img_milestone_cache = {}
        self.img_milestone_root_dir = "milestones/img"
        # caches and helpers for syntax-aware similarity
        self._sc_milestone_slot_cache: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self._spacy_nlp = None
        self._spacy_nlp_failed = False
        self._syntax_similarity_weights = {"verb": 0.3, "object": 0.6, "modifier": 0.1}

        # if not self.config.milestone_reward.load_path:
        #     milestone_dict_path = 'milestones/milestone_dict.json'
        # else:
        #     milestone_dict_path = self.config.milestone_reward.load_path
        # with open(milestone_dict_path, 'r') as f:
        #     self.milestone_dict = json.load(f)
        # self._sc_milestone_embed_cache = {}
        # if self.config.milestone_reward.hit_num_load_path:
        #     with open(self.config.milestone_reward.hit_num_load_path, 'r') as f:
        #         self.milestone_history_hit_counts = json.load(f)
        # else:
        #     self.milestone_history_hit_counts={}
        #     for task in self.milestone_dict:
        #         for milestone_index in range(len(self.milestone_dict[task])):
        #             self.milestone_history_hit_counts[(task, milestone_index)] = {"hit_count": 0, "threshold": self.config.milestone_reward.threshold}

        if not self.config.milestone_reward.load_path:
            milestone_dict_path = 'milestones/milestone_dict.json'
            self.milestone_history_hit_counts={}
            self.milestone_dict={}
            self._sc_milestone_embed_cache = {}
        else:
            milestone_dict_path = self.config.milestone_reward.load_path
            with open(milestone_dict_path, 'r') as f:
                self.milestone_dict = json.load(f)
            self._sc_milestone_embed_cache = {}
            if self.config.milestone_reward.hit_num_load_path:
                with open(self.config.milestone_reward.hit_num_load_path, 'r') as f:
                    self.milestone_history_hit_counts = json.load(f)
            else:
                self.milestone_history_hit_counts={}
                for task in self.milestone_dict:
                    for milestone_index in range(len(self.milestone_dict[task])):
                        self.milestone_history_hit_counts[(task, milestone_index)] = {"hit_count": 0, "threshold": self.config.milestone_reward.threshold}
            
        
        self.milestone_history_hit_counts_path = self.config.trainer.default_local_dir
        if self.milestone_history_hit_counts_path:
            os.makedirs(self.milestone_history_hit_counts_path, exist_ok=True)
            self.milestone_similarity_log_path = os.path.join(
                self.milestone_history_hit_counts_path, "milestone_similarity_matches.json"
            )
        else:
            self.milestone_similarity_log_path = os.path.join(
                os.getcwd(), "milestone_similarity_matches.json"
            )

        self.task_difficulty_dict = {}
        if self.config.Difficulty_factor.load_path:
            with open(self.config.Difficulty_factor.load_path, 'r') as f:
                self.task_difficulty_dict = json.load(f)


    def _ensure_spacy_model(self) -> bool:
        if self._spacy_nlp is not None:
            return True
        if self._spacy_nlp_failed:
            return False
        if spacy is None:
            self._spacy_nlp_failed = True
            return False
        try:
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._spacy_nlp = None
            self._spacy_nlp_failed = True
            return False
        return True

    @staticmethod
    def _unique_keep_order(items: List[str]) -> List[str]:
        seen = set()
        results: List[str] = []
        for item in items:
            normalized = item.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(item.strip())
        return results

    def _extract_sentence_slots(self, sentence: str) -> Dict[str, str]:
        slots = {"verb": "", "object": "", "modifier": ""}
        if not sentence:
            return slots

        if self._ensure_spacy_model():
            try:
                doc = self._spacy_nlp(sentence)
            except Exception:
                doc = None
            if doc is not None:
                verbs: List[str] = []
                objects: List[str] = []
                modifiers: List[str] = []
                for token in doc:
                    if token.pos_ == "VERB":
                        lemma = token.lemma_.lower()
                        if token.dep_ == "ROOT":
                            verbs.insert(0, lemma)
                        else:
                            verbs.append(lemma)
                    if token.dep_ in {"dobj", "pobj", "obj", "attr"} or (
                        token.pos_ in {"NOUN", "PROPN"} and token.dep_ in {"compound", "dative", "nsubj", "nsubjpass"}
                    ):
                        subtree_tokens = [t.text for t in token.subtree if not t.is_space]
                        if subtree_tokens:
                            objects.append(" ".join(subtree_tokens))
                    if token.dep_ in {"advmod", "amod"} or token.pos_ in {"ADV", "ADJ"}:
                        modifiers.append(token.lemma_.lower())

                if not objects:
                    try:
                        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                    except Exception:
                        noun_chunks = []
                    objects.extend(noun_chunks)

                verbs = self._unique_keep_order(verbs)
                objects = self._unique_keep_order(objects)
                modifiers = self._unique_keep_order(modifiers)

                if verbs:
                    slots["verb"] = " ; ".join(verbs)
                if objects:
                    slots["object"] = " ; ".join(objects)
                if modifiers:
                    slots["modifier"] = " ; ".join(modifiers)

                if slots["object"] and slots["verb"]:
                    return slots

        tokens = [token for token in sentence.split() if token]
        if tokens:
            slots["verb"] = tokens[0].lower()
        if len(tokens) > 1:
            filtered = [t for t in tokens[1:] if t.lower() not in {"the", "a", "an", "to", "please"}]
            if filtered:
                slots["object"] = " ".join(filtered[:6])
        modifiers = [t for t in tokens if t.lower().endswith("ly")]
        if modifiers:
            slots["modifier"] = " ".join(modifiers[:4])
        return slots

    def _embed_slot_values(self, slot_texts: Dict[str, str]) -> Dict[str, Optional[torch.Tensor]]:
        slot_embeddings: Dict[str, Optional[torch.Tensor]] = {}
        for key, value in slot_texts.items():
            value = value.strip()
            if value:
                embedding = self.bert_embedder.embed([value])[0].detach().cpu()
                slot_embeddings[key] = embedding
            else:
                slot_embeddings[key] = None
        return slot_embeddings

    def _get_milestone_slot_embeddings(self, task: str) -> List[Dict[str, Any]]:
        if task in self._sc_milestone_slot_cache:
            return self._sc_milestone_slot_cache[task]
        milestone_sentences = self.milestone_dict.get(task, [])
        slot_entries: List[Dict[str, Any]] = []
        for sentence in milestone_sentences:
            slot_texts = self._extract_sentence_slots(sentence)
            slot_embeddings = self._embed_slot_values(slot_texts)
            slot_entries.append(
                {
                    "sentence": sentence,
                    "slots": slot_texts,
                    "embeddings": slot_embeddings,
                }
            )
        self._sc_milestone_slot_cache[task] = slot_entries
        return slot_entries

    def _compute_slot_similarity(self, milestone_slot_entry: Dict[str, Any], action_slot_entry: Dict[str, Any]) -> float:
        milestone_embeddings: Dict[str, Optional[torch.Tensor]] = milestone_slot_entry.get("embeddings", {})
        milestone_texts: Dict[str, str] = milestone_slot_entry.get("slots", {})
        action_embeddings: Dict[str, Optional[torch.Tensor]] = action_slot_entry.get("embeddings", {})
        action_texts: Dict[str, str] = action_slot_entry.get("slots", {})

        total_weight = 0.0
        score = 0.0
        for slot, weight in self._syntax_similarity_weights.items():
            ms_emb = milestone_embeddings.get(slot)
            ac_emb = action_embeddings.get(slot)
            slot_score: Optional[float] = None
            if ms_emb is not None and ac_emb is not None:
                slot_score = float(
                    torch.nn.functional.cosine_similarity(
                        ms_emb.unsqueeze(0), ac_emb.unsqueeze(0), dim=-1
                    ).item()
                )
            else:
                ms_text = milestone_texts.get(slot, "").strip().lower()
                ac_text = action_texts.get(slot, "").strip().lower()
                if ms_text and ac_text:
                    slot_score = 1.0 if ms_text == ac_text else 0.0
            if slot_score is None:
                continue
            total_weight += weight
            score += weight * max(min(slot_score, 1.0), -1.0)
        if total_weight == 0.0:
            return 0.0
        return score / total_weight


    def _create_envs(self):
        max_envs = self.config.env.max_envs
        src = self.config.env.src
        assert len(max_envs) == len(src)

        train_batch_size = self.config.data.train_batch_size
        n = self.config.actor_rollout_ref.rollout.n
        num_envs = train_batch_size * n

        self.env_workers = []
        for i in range(len(max_envs)):
            self.env_workers += [
                EnvWorker.remote(worker_id=f"env_worker_{uuid.uuid4().hex}", config=self.config, src=src[i]) for _ in range(int(max_envs[i]))
            ]

    def _get_alive_env_works(self):
        alives = ray.get([worker.is_alive.remote() for worker in self.env_workers])
        # pprint(f"# of alive environments: {sum(alives)}")
        return [env for env, is_alive in zip(self.env_workers, alives) if is_alive]

    def _get_alive_env_batch(self, batch_task, batch_env_output, batch_env_workers):
        available_batch_task = []
        available_batch_env_output = []
        available_batch_env_workers = []
        available_indices = []
        for idx, (task, env_output, env_worker) in enumerate(
            zip(batch_task, batch_env_output, batch_env_workers)
        ):
            if not env_output["is_alive"]:
                continue
            available_indices.append(idx)
            available_batch_task.append(task)
            available_batch_env_output.append(env_output)
            available_batch_env_workers.append(env_worker)
        return available_batch_task, available_batch_env_output, available_batch_env_workers

    def _get_unfinished_env_batch(self, batch_task, batch_env_output, batch_env_workers):
        unfinished_batch_task = []
        unfinished_batch_env_output = []
        unfinished_batch_env_workers = []
        for task, env_output, env_worker in zip(batch_task, batch_env_output, batch_env_workers):
            if env_output["is_done"]:
                continue
            unfinished_batch_task.append(task)
            unfinished_batch_env_output.append(env_output)
            unfinished_batch_env_workers.append(env_worker)
        return unfinished_batch_task, unfinished_batch_env_output, unfinished_batch_env_workers

    def _get_finished_env_batch(self, batch_task, batch_env_output, batch_env_workers):
        finished_batch_task = []
        finished_batch_env_output = []
        finished_batch_env_workers = []
        for task, env_output, env_worker in zip(batch_task, batch_env_output, batch_env_workers):
            if env_output["is_done"]:
                finished_batch_task.append(task)
                finished_batch_env_output.append(env_output)
                finished_batch_env_workers.append(env_worker)
        return finished_batch_task, finished_batch_env_output, finished_batch_env_workers

    def _init_envs(self, batch_task, is_training: bool = True):
        num_env_workers = len(batch_task)
        alive_env_workers = self._get_alive_env_works()
        assert (
            len(alive_env_workers) >= num_env_workers
        ), f"# of alive env worker {len(alive_env_workers)} is not enough (< train_batch_size * rollout_n {len(batch_task)})."

        batch_env_workers = alive_env_workers[:num_env_workers]
        batch_env_output = ray.get(
            [worker.init_task.remote(task=t, is_training=is_training) for worker, t in zip(batch_env_workers, batch_task)],
            timeout=INIT_TASK_TIMEOUT,
        )
        return batch_task, batch_env_output, batch_env_workers

    def _reset_envs(self):
        ray.get([worker.reset.remote() for worker in self.env_workers])

    def _close_envs(self):
        ray.get([worker.release.remote() for worker in self.env_workers])

    def _generate_rollout_batch(self, batch_env_output=None, messages=None, is_training=False):
        if batch_env_output is not None:
            messages = []
            for _i, x in enumerate(batch_env_output):
                idx = 0
                single_msgs = x["messages"]
                for i in range(len(single_msgs) - 1, -1, -1):
                    if single_msgs[i]["role"] == "system":
                        idx = i
                        break
                messages.append(single_msgs[idx:])

        if is_training and self.config.strategy.use_human_helps:
            messages = get_batch_human_helps(
                batch_messages=messages,
                task_ids=[x["task_id"] for x in batch_env_output],
                human_helps_ratio=self.config.strategy.use_human_helps_ratio,
                openai_api_key=self.config.strategy.openai_api_key,
                openai_api_base=self.config.strategy.openai_api_base,
                helper_model_name=self.config.strategy.helper_model_name,
                max_tokens=self.config.actor_rollout_ref.rollout.response_length,
            )
        elif is_training and self.config.strategy.use_reflection:
            messages = get_batch_reflection(
                batch_messages=messages,
                batch_history_messages=[x["messages"] for x in batch_env_output], 
                batch_goals=[x["task"]["goal"] for x in batch_env_output],
                openai_api_key=self.config.strategy.openai_api_key,
                openai_api_base=self.config.strategy.openai_api_base,
                teacher_model_name=self.config.strategy.helper_model_name,
                max_tokens=512,
            )
        dataset = MessagesDataset(
            messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.truncation,
            use_human_helps=self.config.strategy.use_human_helps
        )

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(lambda x: dataset[x], range(len(dataset))))

        batch_dict = collate_fn(batch_dict)
        batch: DataProto = DataProto.from_single_dict(batch_dict)  # auto_padding=False

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        if is_training and (self.config.strategy.use_human_helps or self.config.strategy.use_reflection):
            batch_keys_to_pop.append("attention_mask_wo_human_helps")
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        return batch, gen_batch

    def _validate(self):
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        batch_task_all = []
        n = self.config.actor_rollout_ref.rollout.val_kwargs.n
        for test_data in self.val_dataloader:
            batch_task_all += [deepcopy(t) for t in test_data for _ in range(n)]
        random.shuffle(batch_task_all)
        batch_task_all = [t | {"index": i, "num_tries": 0} for i, t in enumerate(batch_task_all)]

        while len(batch_task_all) > 0:
            self._reset_envs()
            num_alive_workers = len(self._get_alive_env_works())
            pprint(f"# of alive environments: {num_alive_workers}")
            pprint(f"# of remaining tasks (rollouts): {len(batch_task_all)}")
            if num_alive_workers == 0:
                break

            for t in batch_task_all[: num_alive_workers]:
                t["num_tries"] += 1

            finished_test_batch_task = []
            finished_test_batch_env_workers = []

            batch_task, batch_env_output, batch_env_workers = self._init_envs(
                batch_task=batch_task_all[: num_alive_workers], is_training=False
            )
            for _ in range(self.config.env.val_max_steps):
                # alive envs
                batch_task, batch_env_output, batch_env_workers = self._get_alive_env_batch(
                    batch_task, batch_env_output, batch_env_workers
                )
                if len(batch_env_workers) == 0:
                    break
                # finished
                finished_batch_task, _, finished_batch_env_workers = self._get_finished_env_batch(
                    batch_task, batch_env_output, batch_env_workers
                )
                finished_test_batch_task.extend(finished_batch_task)
                finished_test_batch_env_workers.extend(finished_batch_env_workers)
                # unfinished
                batch_task, batch_env_output, batch_env_workers = self._get_unfinished_env_batch(
                    batch_task, batch_env_output, batch_env_workers
                )
                if len(batch_env_workers) == 0:
                    break

                _, gen_batch = self._generate_rollout_batch(batch_env_output=batch_env_output, is_training=False)
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }

                # pad to be divisible by dp_size
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                    gen_batch, self.actor_rollout_wg.world_size
                )
                if not self.async_rollout_mode:
                    gen_batch_padded_output = self.actor_rollout_wg.generate_sequences(
                        gen_batch_padded
                    )
                else:
                    self.async_rollout_manager.wake_up()
                    gen_batch_padded_output = self.async_rollout_manager.generate_sequences(
                        gen_batch_padded
                    )
                    self.async_rollout_manager.sleep()

                # unpad
                gen_batch_output = unpad_dataproto(gen_batch_padded_output, pad_size=pad_size)

                # next action
                response_text_batch = self.tokenizer.batch_decode(
                    gen_batch_output.batch["responses"],
                    skip_special_tokens=True,
                )
                history_infos_batch = [x["history_infos"] for x in batch_env_output]
                batch_env_output = ray.get(
                    [
                        worker.step.remote(action_text, history_infos)
                        for worker, action_text, history_infos in zip(batch_env_workers, response_text_batch, history_infos_batch)
                    ]
                )

            # last step
            # alive envs
            batch_task, batch_env_output, batch_env_workers = self._get_alive_env_batch(
                batch_task, batch_env_output, batch_env_workers
            )
            # finished
            finished_batch_task, _, finished_batch_env_workers = self._get_finished_env_batch(
                batch_task, batch_env_output, batch_env_workers
            )
            finished_test_batch_task.extend(finished_batch_task)
            finished_test_batch_env_workers.extend(finished_batch_env_workers)
            # unfinished
            batch_task, batch_env_output, batch_env_workers = self._get_unfinished_env_batch(
                batch_task, batch_env_output, batch_env_workers
            )

            # evaluate using environment rule-based reward_function
            finished_batch_scores = ray.get(
                [w.evaluate.remote() for w in finished_test_batch_env_workers]
            )
            unfinished_batch_scores = [0] * len(batch_env_workers)
            batch_scores = finished_batch_scores + unfinished_batch_scores
            pprint(f"batch scores: {list(zip([t['task'] for t in finished_test_batch_task + batch_task], batch_scores))}")
            with open(f"{self.config.trainer.validation_data_dir}/val_logs_{self.global_steps}.jsonl", 'a', encoding="utf-8") as outfile:
                for t, score in zip(finished_test_batch_task + batch_task, batch_scores):
                    outfile.write(json.dumps({"task_id": t["task_id"], "score": score}, ensure_ascii=False) + "\n")
            sample_scores.extend(batch_scores)
            sample_inputs.extend(finished_test_batch_task + batch_task)
            batch_messages = ray.get(
                [
                    w.history_messages.remote()
                    for w in (finished_test_batch_env_workers + batch_env_workers)
                ]
            )
            sample_outputs.extend([[_ for _ in x if _["role"] == "assistant"] for x in batch_messages])

            reward_extra_infos_dict["reward"].extend(batch_scores)

            rollouted_indexs = set([t["index"] for t in finished_test_batch_task + batch_task])
            failed_sample_infos = [t for t in batch_task_all if t["index"] not in rollouted_indexs and t["num_tries"] >= self.config.env.val_max_tries_per_rollout]
            if failed_sample_infos:
                pprint(failed_sample_infos)
            batch_task_all = [t for t in batch_task_all if t["index"] not in rollouted_indexs and t["num_tries"] < self.config.env.val_max_tries_per_rollout]

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(
                sample_scores
            ), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        reward_score = torch.tensor(sample_scores).mean().item()
        pprint(f"val/reward_score: {reward_score}")

        return {"val/reward_score": reward_score}

    def _validate_static(self):
        val_batch_size = self.config.actor_rollout_ref.rollout.max_num_seqs
        benchmark_responses = []
        for i in range(0, len(self.benchmark_messages), val_batch_size):
            _, gen_batch = self._generate_rollout_batch(messages=self.benchmark_messages[i: i + val_batch_size], is_training=False)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                gen_batch, self.actor_rollout_wg.world_size
            )
            if not self.async_rollout_mode:
                gen_batch_padded_output = self.actor_rollout_wg.generate_sequences(
                    gen_batch_padded
                )
            else:
                self.async_rollout_manager.wake_up()
                gen_batch_padded_output = self.async_rollout_manager.generate_sequences(
                    gen_batch_padded
                )
                self.async_rollout_manager.sleep()
            gen_batch_output = unpad_dataproto(gen_batch_padded_output, pad_size=pad_size)

            response_text_batch = self.tokenizer.batch_decode(
                gen_batch_output.batch["responses"],
                skip_special_tokens=True,
            )
            benchmark_responses += response_text_batch
        static_val_result = benchmark_evaluate(
            self.benchmark_infos, 
            benchmark_responses, 
            f"{self.config.trainer.validation_data_dir}/static", 
            self.global_steps
        )
        return static_val_result


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._create_envs()

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            # static_val_metrics = self._validate_static()
            # pprint(f"Initial static validation metrics: {static_val_metrics}")
            # logger.log(data=static_val_metrics, step=self.global_steps)

            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # use human helps
        if self.config.strategy.use_human_helps:
            with open(self.config.strategy.human_helps_file, 'r') as infile:
                self.human_helps = json.load(infile)

        # use replay
        if self.config.strategy.use_replay:
            if self.config.strategy.replay_load_path:
                replay_buffer = DataProto.load_from_disk(self.config.strategy.replay_load_path)
                replay_buffer.batch["accuracy_rewards"] = torch.tensor([1.0 for _ in replay_buffer.batch["accuracy_rewards"]], dtype=torch.float32)
                pprint(f"Loading replay buffer {len(replay_buffer)} from: {self.config.strategy.replay_load_path}")

                buffer_uids = replay_buffer.non_tensor_batch["uid"].tolist()
                buffer_num_steps = replay_buffer.batch["num_steps"].tolist()
                pairs = list(zip(buffer_uids, buffer_num_steps))
                self.min_num_steps_map = {u: min([_[1] for _ in pairs if _[0] == u]) for u in set(buffer_uids)}
            else:
                replay_buffer = DataProto.from_single_dict({})
                self.min_num_steps_map = {}
        # self.task_difficulty_dict = {}
        train_data = []
        for batch_dict in self.train_dataloader:
            train_data += batch_dict
        self.mean_accuracy_reward_map = {x["task_id"]: 0.0 for x in train_data}

        for epoch in range(self.config.trainer.total_epochs):
            if epoch == 0:
                train_dataloader = train_data
            else:
                sorted_train_data = sorted(
                    [x | {"mean_accuracy_reward": self.mean_accuracy_reward_map[x["task_id"]]} for x in train_data], 
                    key=lambda x: x["mean_accuracy_reward"], reverse=True
                )
                if epoch % 2 == 0:
                    train_dataloader = sorted_train_data
                else:
                    sorted_train_data = [x for x in sorted_train_data if 0 < x["mean_accuracy_reward"] < 1]
                    train_dataloader = []
                    for i in range(0, len(sorted_train_data), self.config.data.train_batch_size):
                        bth = sorted_train_data[i: i + self.config.data.train_batch_size]
                        train_dataloader += bth + bth

            with open(f"{self.config.trainer.rollout_data_dir}/train_data_{epoch}.jsonl", 'a', encoding="utf-8") as outfile:
                for i, x in enumerate(train_dataloader):
                    outfile.write(json.dumps({"index": i, "uid": x["task_id"], "mean_accuracy_reward": self.mean_accuracy_reward_map[x["task_id"]]}, ensure_ascii=False) + "\n")
                    
            for bidx in range(0, len(train_dataloader), self.config.data.train_batch_size):
                batch_dict = train_dataloader[bidx: bidx + self.config.data.train_batch_size]
                batch_size = len(batch_dict)
                metrics = {}
                timing_raw = {}

                n = self.config.actor_rollout_ref.rollout.n
                batch_task_all = [deepcopy(t) for t in batch_dict for _ in range(n)]
                random.shuffle(batch_task_all)
                batch_task_all = [t | {"index": i, "num_tries": 0} for i, t in enumerate(batch_task_all)]
                
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        batchs = []
                        while len(batch_task_all) > 0:
                            self._reset_envs()
                            num_alive_workers = len(self._get_alive_env_works())
                            pprint(f"# of alive environments: {num_alive_workers}")
                            if num_alive_workers == 0:
                                break
                            
                            for t in batch_task_all[: num_alive_workers]:
                                t["num_tries"] += 1
                            with _timer("init_envs", timing_raw):
                                batch_task, batch_env_output, batch_env_workers = self._init_envs(
                                    batch_task=batch_task_all[: num_alive_workers], is_training=True
                                )
                                
                            finished_train_batch_task = []
                            finished_train_batch_env_output = []
                            finished_train_batch_env_workers = []

                            step_batchs = []
                            for _ in range(self.config.env.max_steps):
                                # alive envs
                                batch_task, batch_env_output, batch_env_workers = (
                                    self._get_alive_env_batch(
                                        batch_task, batch_env_output, batch_env_workers
                                    )
                                )
                                if len(batch_env_workers) == 0:
                                    break
                                # finished
                                (
                                    finished_batch_task,
                                    finished_batch_env_output,
                                    finished_batch_env_workers,
                                ) = self._get_finished_env_batch(
                                    batch_task, batch_env_output, batch_env_workers
                                )
                                finished_train_batch_task.extend(finished_batch_task)
                                finished_train_batch_env_output.extend(finished_batch_env_output)
                                finished_train_batch_env_workers.extend(finished_batch_env_workers)
                                # unfinished
                                batch_task, batch_env_output, batch_env_workers = (
                                    self._get_unfinished_env_batch(
                                        batch_task, batch_env_output, batch_env_workers
                                    )
                                )
                                if len(batch_env_workers) == 0:
                                    break
                                
                                with _timer("generate_rollout_batch", timing_raw):
                                    step_batch, gen_batch = self._generate_rollout_batch(
                                        batch_env_output=batch_env_output, is_training=True
                                    )

                                gen_batch.meta_info = {
                                    "eos_token_id": self.tokenizer.eos_token_id,
                                    "pad_token_id": self.tokenizer.pad_token_id,
                                    "recompute_log_prob": True,
                                    "output_attentions": True,
                                    "is_repeated": True, 
                                }                                
                                # pad to be divisible by dp_size
                                gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                                    gen_batch, self.actor_rollout_wg.world_size
                                )
                                if not self.async_rollout_mode:
                                    # verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
                                    gen_batch_padded_output = self.actor_rollout_wg.generate_sequences(  
                                        gen_batch_padded
                                    )
                                else:
                                    self.async_rollout_manager.wake_up()
                                    gen_batch_padded_output = (
                                        self.async_rollout_manager.generate_sequences(gen_batch_padded)
                                    )
                                    self.async_rollout_manager.sleep()

                                # unpad
                                gen_batch_output = unpad_dataproto(
                                    gen_batch_padded_output, pad_size=pad_size
                                )
                                # Retrieve attention matrices via Hugging Face (head-averaged, last layer)
                                
                               
                                if "rollout_log_probs" in gen_batch_output.batch:

                                
                                    rollout_log_probs = gen_batch_output.batch["rollout_log_probs"]  # [B, T]
                                    base_mask = (rollout_log_probs != -1)  # [B, T]
                                    valid_count_all = base_mask.sum(dim=-1).float()
                                    responses_ids = gen_batch_output.batch["responses"]  # [B, T]
                                    response_text_batch_local = self.tokenizer.batch_decode(
                                        responses_ids, skip_special_tokens=True
                                    )

                                    confidence_list = []
                                    conf_entropy_list = []
                                    action_confidence_list = []
                                    kl_uniform_list = []
                                    
                                    for i, full_text in enumerate(response_text_batch_local):
                                        # default: use old mask (all response tokens)
                                        token_mask_selected = base_mask[i].clone()
                                        
                                        
                                        # Calculate thought part confidence and entropy
                                        thought_confidence = None
                                        thought_entropy = None
                                        KL_uniform = torch.tensor(0.0)
                                        
                                        try:
                                            selected_char_spans: List[Tuple[int, int]] = []
                                            action_span: Optional[Tuple[int, int]] = None
                                            ans_start = full_text.find("<answer>")
                                            ans_end = full_text.find("</answer>", ans_start + 8) if ans_start != -1 else -1
                                            thought_region_end: Optional[int] = ans_start if ans_start != -1 else None

                                            # Calculate thought part (before <answer> or before Action:)
                                            if ans_start != -1:
                                                thought_text = full_text[:ans_start]
                                                if thought_text.strip():  # only if there's actual thought content
                                                    try:
                                                        enc_thought = self.tokenizer(
                                                            full_text,
                                                            add_special_tokens=False,
                                                            return_offsets_mapping=True,
                                                        )
                                                        offsets_thought = enc_thought.get("offset_mapping", None)
                                                        
                                                        if offsets_thought is not None and abs(len(offsets_thought) - valid_count_all[i].item()) <= 10:
                                                            thought_mask = torch.zeros_like(base_mask[i])
                                                            for tok_idx, (s, e) in enumerate(offsets_thought):
                                                                if e <= s:
                                                                    continue
                                                                # token overlaps with thought part (before <answer>)
                                                                if not (e <= 0 or s >= ans_start):
                                                                    thought_mask[tok_idx] = True
                                                            
                                                            thought_final_mask = base_mask[i] & thought_mask
                                                            if thought_final_mask.sum().item() > 0:
                                                                lp = rollout_log_probs[i]
                                                                thought_valid_count = thought_final_mask.sum().float().clamp(min=1.0)
                                                                thought_entropy_val = (lp * thought_final_mask * torch.exp(lp))

                                                                thought_prob= torch.exp(lp)
                                                                thought_prob = thought_prob * thought_final_mask
                                                                thought_prob = thought_prob / thought_prob.sum(dim=-1, keepdim=True)
                                                                KL_uniform = thought_prob * (torch.log(thought_prob.clamp(min=1e-10)) + math.log(thought_valid_count))
                                                                KL_uniform = KL_uniform.sum(dim=-1)
                                                                

                                                                max_thought_entropy = torch.max(thought_entropy_val, dim=-1).values
                                                                thought_entropy_val = thought_entropy_val.sum() / thought_valid_count
                                                                thought_avg_log_prob = (lp * thought_final_mask).sum() / thought_valid_count
                                                                
                                                                thought_entropy = thought_entropy_val.detach().cpu().item()
                                                                thought_confidence = math.exp(thought_avg_log_prob.detach().cpu().item())
                                                               
                                                                
                                                    except Exception as e:
                                                        pass  # thought calculation failed, leave as None
                                            if ans_start != -1 and ans_end != -1:
                                                answer_segment = full_text[ans_start + 8 : ans_end]
                                                patterns = [
                                                    r'"action"\s*:\s*"([^"]+)"',  # mobile_use.action
                                                    r'"coordinate"\s*:\s*\[([^\]]+)\]',  # mobile_use.coordinate
                                                    r'"coordinate2"\s*:\s*\[([^\]]+)\]',  # mobile_use.coordinate2
                                                    r'"button"\s*:\s*"([^"]+)"',  # mobile_use.button (system_button)
                                                    r'"text"\s*:\s*"([^"]+)"',  # mobile_use.text/type/open
                                                    r'"status"\s*:\s*"([^"]+)"',  # mobile_use.terminate.status
                                                    r'"action_type"\s*:\s*"([^"]+)"',  # agent.action_type
                                                    r'"x"\s*:\s*(-?\d+(?:\.\d+)?)',  # agent.x
                                                    r'"y"\s*:\s*(-?\d+(?:\.\d+)?)',  # agent.y
                                                    r'"direction"\s*:\s*"([^"]+)"',  # agent.scroll.direction
                                                    r'"app_name"\s*:\s*"([^"]+)"',  # agent.open_app.app_name
                                                    r'"goal_status"\s*:\s*"([^"]+)"',  # agent.status.goal_status
                                                ]
                                                for pat in patterns:
                                                    for m in re.finditer(pat, answer_segment, flags=re.DOTALL):
                                                        s1, e1 = m.span(1)
                                                        selected_char_spans.append((ans_start + s1, ans_start + e1))
                                            elif self.model_variant == "tars":
                                                action_match = re.search(r"Action:\s*(.*)", full_text, re.DOTALL)
                                                if action_match:
                                                    action_body = action_match.group(1)
                                                    blank_match = re.search(r"\n\s*\n", action_body)
                                                    action_start = action_match.start(1)
                                                    if blank_match:
                                                        action_end = action_start + blank_match.start()
                                                    else:
                                                        action_end = action_match.end(1)
                                                    selected_char_spans.append((action_start, action_end))
                                                    action_span = (action_start, action_end)
                                                    thought_region_end = action_match.start()

                                            if thought_region_end is not None and thought_region_end > 0:
                                                thought_text = full_text[:thought_region_end]
                                                if thought_text.strip():
                                                    try:
                                                        enc_thought = self.tokenizer(
                                                            full_text,
                                                            add_special_tokens=False,
                                                            return_offsets_mapping=True,
                                                        )
                                                        offsets_thought = enc_thought.get("offset_mapping", None)
                                                        if offsets_thought is not None and abs(len(offsets_thought) - valid_count_all[i].item()) <= 10:
                                                            thought_mask = torch.zeros_like(base_mask[i])
                                                            for tok_idx, (s, e) in enumerate(offsets_thought):
                                                                if e <= s:
                                                                    continue
                                                                if not (e <= 0 or s >= thought_region_end):
                                                                    thought_mask[tok_idx] = True
                                                            thought_final_mask = base_mask[i] & thought_mask
                                                            if thought_final_mask.sum().item() > 0:
                                                                lp = rollout_log_probs[i]
                                                                thought_valid_count = thought_final_mask.sum().float().clamp(min=1.0)
                                                                thought_entropy_val = (lp * thought_final_mask * torch.exp(lp))
                                                                thought_prob = torch.exp(lp)
                                                                thought_prob = thought_prob * thought_final_mask
                                                                thought_prob = thought_prob / thought_prob.sum(dim=-1, keepdim=True)
                                                                KL_uniform = thought_prob * (torch.log(thought_prob.clamp(min=1e-10)) + math.log(thought_valid_count))
                                                                KL_uniform = KL_uniform.sum(dim=-1)
                                                                thought_entropy_val = thought_entropy_val.sum() / thought_valid_count
                                                                thought_avg_log_prob = (lp * thought_final_mask).sum() / thought_valid_count
                                                                thought_entropy = thought_entropy_val.detach().cpu().item()
                                                                thought_confidence = math.exp(thought_avg_log_prob.detach().cpu().item())
                                                    except Exception:
                                                        pass

                                            if selected_char_spans:
                                                enc = self.tokenizer(
                                                    full_text,
                                                    add_special_tokens=False,
                                                    return_offsets_mapping=True,
                                                )
                                                offsets = enc.get("offset_mapping", None)

                                                if offsets is not None and abs(len(offsets) - valid_count_all[i].item()) <= 10:
                                                    tmp_mask = torch.zeros_like(base_mask[i])
                                                    for tok_idx, (s, e) in enumerate(offsets):
                                                        if e <= s:
                                                            continue
                                                        for (cs, ce) in selected_char_spans:
                                                            if not (e <= cs or s >= ce):
                                                                tmp_mask[tok_idx] = True
                                                                break
                                                    if tmp_mask.sum().item() > 0:
                                                        token_mask_selected = tmp_mask
                                                elif ans_start != -1 and ans_end != -1:
                                                    enc_ans = self.tokenizer(
                                                        full_text[ans_start:ans_end],
                                                        add_special_tokens=False,
                                                        return_offsets_mapping=False,
                                                    )
                                                    ans_token_count = len(enc_ans["input_ids"]) if "input_ids" in enc_ans else 0
                                                    token_mask_selected = torch.zeros_like(base_mask[i])
                                                    token_mask_selected[-ans_token_count:] = True
                                        except Exception:
                                            # any failure, fallback to old logic (all response tokens)
                                            token_mask_selected = base_mask[i]
    
                                        # AND with base mask, ensure only valid tokens are counted
                                        final_mask = base_mask[i] & token_mask_selected

                                        # if selected token is empty, fallback to only <answer> tokens; if still empty, fallback to all response tokens
                                        if final_mask.sum().item() == 0:
                                            try:
                                                if ans_start != -1 and ans_end != -1:
                                                    enc_full = self.tokenizer(
                                                        full_text,
                                                        add_special_tokens=False,
                                                        return_offsets_mapping=True,
                                                    )
                                                    offsets_full = enc_full.get("offset_mapping", None)
                                                    if offsets_full is not None and len(offsets_full) == responses_ids.size(1):
                                                        token_mask_answer = torch.zeros_like(base_mask[i])
                                                        for tok_idx, (s, e) in enumerate(offsets_full):
                                                            if e <= s:
                                                                continue
                                                            if not (e <= ans_start or s >= ans_end):
                                                                token_mask_answer[tok_idx] = True
                                                        final_mask = base_mask[i] & token_mask_answer
                                            except Exception:
                                                pass

                                        if final_mask.sum().item() == 0 and action_span is not None:
                                            try:
                                                enc_full = self.tokenizer(
                                                    full_text,
                                                    add_special_tokens=False,
                                                    return_offsets_mapping=True,
                                                )
                                                offsets_full = enc_full.get("offset_mapping", None)
                                                if offsets_full is not None and len(offsets_full) == responses_ids.size(1):
                                                    token_mask_action = torch.zeros_like(base_mask[i])
                                                    for tok_idx, (s, e) in enumerate(offsets_full):
                                                        if e <= s:
                                                            continue
                                                        if not (e <= action_span[0] or s >= action_span[1]):
                                                            token_mask_action[tok_idx] = True
                                                    final_mask = base_mask[i] & token_mask_action
                                            except Exception:
                                                pass

                                        if final_mask.sum().item() == 0:
                                            # still empty, fallback to old logic
                                            final_mask = base_mask[i]

                                        # calculate confidence entropy and average log prob for action part
                                        lp = rollout_log_probs[i]
   
                                        valid_count = final_mask.sum().float().clamp(min=1.0)
                                        conf_entropy = (lp * final_mask * torch.exp(lp)).sum() / valid_count
                                        avg_log_prob = (lp * final_mask).sum() / valid_count

                                        action_confidence = math.exp(avg_log_prob.detach().cpu().item())
                                        action_entropy = conf_entropy.detach().cpu().item()


                                        # conf_entropy_list.append(action_entropy)
                                        # confidence_list.append(action_confidence)
                                        kl_uniform_list.append(KL_uniform.detach().cpu().item())
                                        conf_entropy_list.append(thought_entropy)
                                        if thought_confidence is not None:
                                            confidence_list.append(thought_confidence)
                                        else:
                                            confidence_list.append(action_confidence)
                                        action_confidence_list.append(action_confidence)

                                    step_batch.non_tensor_batch["confidence_entropy"] = np.array(conf_entropy_list, dtype=object)
                                    step_batch.non_tensor_batch["confidence"] = np.array(confidence_list, dtype=object)
                                    step_batch.non_tensor_batch["action_confidence"] = np.array(action_confidence_list, dtype=object)
                                    step_batch.non_tensor_batch["kl_uniform"] = np.array(kl_uniform_list, dtype=object) 
                                
                                response_text_batch = self.tokenizer.batch_decode(
                                    gen_batch_output.batch["responses"],
                                    skip_special_tokens=True,
                                )
                                history_infos_batch = [x["history_infos"] for x in batch_env_output]
                                with _timer("env_step", timing_raw):
                                    batch_env_output = ray.get(
                                        [
                                            worker.step.remote(action_text, history_infos)
                                            for worker, action_text, history_infos in zip(
                                                batch_env_workers, response_text_batch, history_infos_batch
                                            )
                                        ]
                                    )

                                step_batch.non_tensor_batch["uid"] = np.array([task["task_id"] for task in batch_task], dtype=object)
                                step_batch.non_tensor_batch["worker_id"] = np.array([_["id"] for _ in batch_env_output], dtype=object)
                                step_batch.non_tensor_batch["task"] = np.array([task["task_id"].split('_')[0] for task in batch_task], dtype=object)
                                step_batch = step_batch.union(gen_batch_output)
                                step_batch.batch["format_rewards"] = torch.tensor([float(_["format_reward"]) for _ in batch_env_output], dtype=torch.float32)
                                step_batch.batch["validity_rewards"] = torch.tensor([float(_["validity_reward"]) for _ in batch_env_output], dtype=torch.float32)

                                if self.config.strategy.use_human_helps:
                                    attention_mask_wo_human_helps = gen_batch.batch["attention_mask_wo_human_helps"]
                                    step_batch.batch["attention_mask"][:, : attention_mask_wo_human_helps.size(1)] = attention_mask_wo_human_helps
                                step_batchs.append(step_batch)

                            # last step
                            # alive envs
                            batch_task, batch_env_output, batch_env_workers = self._get_alive_env_batch(
                                batch_task, batch_env_output, batch_env_workers
                            )
                            # finished
                            (
                                finished_batch_task,
                                finished_batch_env_output,
                                finished_batch_env_workers,
                            ) = self._get_finished_env_batch(
                                batch_task, batch_env_output, batch_env_workers
                            )
                            finished_train_batch_task.extend(finished_batch_task)
                            finished_train_batch_env_output.extend(finished_batch_env_output)
                            finished_train_batch_env_workers.extend(finished_batch_env_workers)
                            # unfinished
                            batch_task, batch_env_output, batch_env_workers = (
                                self._get_unfinished_env_batch(
                                    batch_task, batch_env_output, batch_env_workers
                                )
                            )

                            # batch
                            num_finished_samples = len(finished_train_batch_task)
                            num_unfinished_samples = len(batch_task)
                            num_total_samples = min(num_alive_workers, len(batch_task_all))
                            num_failed_samples = (
                                num_total_samples - num_finished_samples - num_unfinished_samples
                            )

                            pprint(
                                f"# of total samples: {num_total_samples}, "
                                f"# of fininshed samples: {num_finished_samples}, "
                                f"# of unfinished samples: {num_unfinished_samples}, "
                                f"# of failed environment workers: {num_failed_samples}"
                            )
                            with open(f"{self.config.trainer.rollout_data_dir}/env_logs.jsonl", 'a', encoding="utf-8") as outfile:
                                outfile.write(json.dumps({"total_samples": num_total_samples, "fininshed samples": num_finished_samples, "unfinished samples": num_unfinished_samples, "failed environment workers": num_failed_samples}, ensure_ascii=False) + "\n")
                            batch_task = finished_train_batch_task + batch_task
                            batch_env_output = finished_train_batch_env_output + batch_env_output
                            batch_env_workers = finished_train_batch_env_workers + batch_env_workers
                            # accuracy reward
                            accuracy_rewards = ray.get(
                                [w.evaluate.remote() for w in finished_train_batch_env_workers]
                            )
                            accuracy_rewards += [0] * num_unfinished_samples
                            assert len(accuracy_rewards) == len(batch_task)

                            if len(step_batchs) > 0:
                                with _timer("prepare_grpo_inputs", timing_raw):
                                    batch = DataProto.concat(step_batchs)
                                    workerId_set = set([env_output["id"] for env_output in batch_env_output])
                                    worker_ids = [i for i, _ in enumerate(batch.non_tensor_batch["worker_id"]) if _ in workerId_set]
                                    batch = batch.select_idxs(worker_ids)

                                    if len(batch) > 0:
                                        accuracy_reward_map = {env_output["id"]: accuracy_reward for env_output, accuracy_reward in zip(batch_env_output, accuracy_rewards)}
                                        num_steps_map = Counter(batch.non_tensor_batch["worker_id"])

                                        _accuracy_rewards = []
                                        _num_steps = []
                                        for worker_id in batch.non_tensor_batch["worker_id"]:
                                            _accuracy_rewards.append(accuracy_reward_map[worker_id])
                                            _num_steps.append(num_steps_map[worker_id])
                                        accuracy_rewards = _accuracy_rewards

                                        batch.batch["accuracy_rewards"] = torch.tensor([float(x) for x in accuracy_rewards], dtype=torch.float32)
                                        batch.batch["num_steps"] = torch.tensor([float(x) for x in _num_steps], dtype=torch.float32)

                                        if self.config.strategy.use_replay and len(replay_buffer) > 0:
                                            _punishs = []
                                            for uid, worker_id, n_steps, acc in zip(batch.non_tensor_batch["uid"].tolist(), batch.non_tensor_batch["worker_id"].tolist(), batch.batch["num_steps"].tolist(), batch.batch["accuracy_rewards"].tolist()):
                                                if acc == 1 and uid in self.min_num_steps_map:
                                                    min_n_steps = self.min_num_steps_map[uid]
                                                    _punishs.append(max(-1, (min_n_steps - n_steps) / min_n_steps))
                                                else:
                                                    _punishs.append(0.0)
                                            batch.batch["punishment"] = torch.tensor([float(x) for x in _punishs], dtype=torch.float32)
                                        else:
                                            batch.batch["punishment"] = torch.tensor([0.0 for _ in range(len(batch))], dtype=torch.float32)
                                        
                                        # Add other necessary fields from history_infos to the batch
                                        tasks = []
                                        goals = []
                                        action_texts = []
                                        action_descs = []
                                        image_paths = []
                                        worker_use_num={}
                                        for worker_id in batch.non_tensor_batch["worker_id"]:
                                            # Find the corresponding env_output
                                            env_output = next((out for out in batch_env_output if out["id"] == worker_id), None)
                                            if env_output:
                                                worker_use_num[worker_id] = worker_use_num.get(worker_id, -1) + 1
                                                history = env_output["history_infos"]
                                                tasks.append(history["task"])
                                                goals.append(history.get("goal", ""))
                       
                                                action_texts.append(history["action_text"][worker_use_num[worker_id]])
                                                action_descs.append(history["action_description"][worker_use_num[worker_id]])
                                                image_paths.append(history["image_path"][worker_use_num[worker_id]+1]) # Image before action
               
                                                
                                                # except:
                                                #     action_texts.append(history["action_text"][-1] if history["action_text"] else "")
                                                #     action_descs.append(history["action_description"][-1] if history["action_description"] else "")
                                                #     image_paths.append(history["image_path"][-1] if history["image_path"] else "") # Image before action


                                        batch.non_tensor_batch["goal"] = goals
                                        batch.non_tensor_batch["action_text"] = action_texts
                                        batch.non_tensor_batch["action_description"] = action_descs
                                        batch.non_tensor_batch["image_path"] = image_paths
                                        batch.non_tensor_batch["task"] = tasks
                                    
                                        timestamp = datetime.now().strftime("%m%d%H%M%S")
                                        batch.non_tensor_batch["worker_id"] = np.array(
                                            [wid + '-' + timestamp for wid in batch.non_tensor_batch["worker_id"]], 
                                            dtype=object
                                        )

                                        batchs.append(batch)

                            rollouted_indexs = set([t["index"] for t in batch_task])
                            failed_sample_infos = [t for t in batch_task_all if t["index"] not in rollouted_indexs and t["num_tries"] >= self.config.env.train_max_tries_per_rollout]
                            if failed_sample_infos:
                                pprint(failed_sample_infos)
                            batch_task_all = [t for t in batch_task_all if t["index"] not in rollouted_indexs and t["num_tries"] < self.config.env.train_max_tries_per_rollout]

                            # reset environment workers
                            # self._reset_envs()
                            # TODO replay

                    if len(batchs) == 0:
                        pprint(f"len(batchs) == 0")
                        continue
                    batch = DataProto.concat(batchs)

                    if len(batch) < self.config.actor_rollout_ref.actor.ppo_mini_batch_size:
                        pprint(f"len(batch) < actor.ppo_mini_batch_size")
                        continue

                    if self.config.strategy.use_replay:
                        # batch.batch.pop("rollout_log_probs")
                        if not self.config.strategy.use_validity_reward:
                            batch.batch.pop("validity_rewards")
                        buffer_wids = replay_buffer.non_tensor_batch["worker_id"].tolist() if len(replay_buffer) > 0 else []
                        buffer_uids = replay_buffer.non_tensor_batch["uid"].tolist() if len(replay_buffer) > 0 else []
                        buffer_insert_time = replay_buffer.non_tensor_batch["insert_time"].tolist() if len(replay_buffer) > 0 else []

                        to_buffer, to_replay = [], []
                        to_buffer_uids = set()
                        uids = batch.non_tensor_batch["uid"].tolist()
                        for uid in set(uids):
                            sub_batch = batch[batch.non_tensor_batch["uid"] == uid]
                            if uid in set(buffer_uids):
                                selected_insert_time, selected_wid = sorted([(buffer_insert_time[i], buffer_wids[i]) for i in range(len(buffer_uids)) if buffer_uids[i] == uid])[-1]
                                to_replay.append(replay_buffer[replay_buffer.non_tensor_batch["worker_id"] == selected_wid])
                                tag = f"{uid}: use replay buffer inserted at {selected_insert_time}."
                            if sub_batch.batch["accuracy_rewards"].sum().item() == 0:
                                # use replay buffer (the latest)
                                if uid in set(buffer_uids):
                                    pass
                                else:
                                    tag = f"{uid}: cannot get successful trajs in replay buffer."
                            else:
                                # insert to replay buffer (the shortest)
                                succ_sub_batch = sub_batch[sub_batch.batch["accuracy_rewards"] == 1]
                                selected_num_steps, selected_wid = sorted(zip(succ_sub_batch.batch["num_steps"], succ_sub_batch.non_tensor_batch["worker_id"]))[0]
                                if uid not in self.min_num_steps_map or selected_num_steps <= self.min_num_steps_map[uid] + self.config.strategy.replay_step_nums_tolerance:
                                    if uid not in self.min_num_steps_map or selected_num_steps < self.min_num_steps_map[uid]:
                                        self.min_num_steps_map[uid] = selected_num_steps
                                    to_buffer.append(succ_sub_batch[succ_sub_batch.non_tensor_batch["worker_id"] == selected_wid])
                                    timestamp = datetime.now().strftime("%m%d%H%M%S")
                                    to_buffer[-1].non_tensor_batch["insert_time"] = np.array([timestamp for _ in range(len(to_buffer[-1]))], dtype=object)
                                    to_buffer_uids.add(uid)
                                    tag = f"{uid}: insert to replay buffer, num_steps={selected_num_steps}"
                                else:
                                    tag = f"{uid}: nothing to insert, shortest traj with num_steps={selected_num_steps} is longer than min_num_steps={self.min_num_steps_map[uid]}+tolerance={self.config.strategy.replay_step_nums_tolerance}"
                            pprint(tag)
                            with open(f"{self.config.trainer.rollout_data_dir}/replay_buffer_logs.jsonl", 'a', encoding="utf-8") as outfile:
                                outfile.write(json.dumps({"step": self.global_steps, "tag": tag}, ensure_ascii=False) + "\n")
                        if len(to_replay) > 0:
                            to_replay = DataProto.concat(to_replay)
                            del to_replay.non_tensor_batch["insert_time"]
                        else:
                            to_replay = None
                        if len(to_buffer) > 0:
                            if len(replay_buffer) > 0:
                                replay_buffer = replay_buffer.select_idxs([i for i, _ in enumerate(buffer_uids) if _ not in to_buffer_uids])
                                replay_buffer = DataProto.concat([replay_buffer] + to_buffer)
                            else:
                                replay_buffer = DataProto.concat(to_buffer)
                            if self.config.strategy.replay_save_path and self.global_steps % 5 == 0:
                                replay_buffer.save_to_disk(self.config.strategy.replay_save_path)
                    else:
                        to_replay = None
                    
                    if len(to_buffer) > 0:
                        cur_buffer = to_buffer[0]
                        milestone_generator = MilestoneGenerator(
                            replay_buffer=cur_buffer,
                            milestone_dict=self.milestone_dict,
                            max_images=5,
                            base_url=self.config.step_judge.address,
                            model_name=self.config.step_judge.model,
                            api_key=self.config.step_judge.api_key,
                        )
                        for goal in set(cur_buffer.non_tensor_batch["goal"]):
                            task = cur_buffer[cur_buffer.non_tensor_batch["goal"] == goal].non_tensor_batch["task"].tolist()[0]
                            self.step_judge.update_ref_task_dict(cur_buffer,task)
                            milestone,_,_,need_update= milestone_generator.generate_from_previous(task)
                            # breakpoint()
                            if need_update:
                                self.milestone_dict[task] = milestone
                                if task in self._sc_milestone_embed_cache:
                                    self._sc_milestone_embed_cache.pop(task, None)
                                self._sc_milestone_embed_cache[task] = self.bert_embedder.embed(milestone)

                                for milestone_index in range(len(milestone)):
                                    self.milestone_history_hit_counts[(task, milestone_index)]={"hit_count": 0, "threshold": self.config.milestone_reward.threshold}

                    if self.config.strategy.downsample_rollout_n:
                        downsample_rollout_n = self.config.strategy.downsample_rollout_n
                        uids = batch.non_tensor_batch["uid"].tolist()
                        worker_ids = batch.non_tensor_batch["worker_id"].tolist()
                        to_replay_uids = set(to_replay.non_tensor_batch["uid"].tolist()) if to_replay is not None else set()
                        tuples = list(zip(uids, worker_ids, list(range(len(uids)))))
                        selected_idxs = []
                        batch_pad = []
                        for uid in set(uids):
                            if uid in to_replay_uids:
                                pick_num = downsample_rollout_n - 1
                            else:
                                pick_num = downsample_rollout_n
                            sub_tuples = [_ for _ in tuples if _[0] == uid]
                            wids = list(set([_[1] for _ in sub_tuples]))
                            pick_wids = random.sample(wids, min(len(wids), pick_num))
                            pick_num -= min(len(wids), pick_num)
                            selected_idxs.extend([_[-1] for _ in sub_tuples if _[1] in set(pick_wids)])
                            for i in range(pick_num):  # pad trajs if any worker failed
                                pick_wid = random.sample(wids, 1)[0]
                                batch_pad.append(batch[batch.non_tensor_batch["worker_id"] == pick_wid])
                                batch_pad[-1].non_tensor_batch["worker_id"] = np.array([f"{_}-{i}" for _ in batch_pad[-1].non_tensor_batch["worker_id"]], dtype=object)
                        batch = batch.select_idxs(selected_idxs)
                        if batch_pad:
                            batch = DataProto.concat([batch] + batch_pad)

                    if to_replay:
                        batch = DataProto.concat([batch, to_replay])

                    pprint(f"rollout_batch_size: {batch.batch.batch_size[0]}")
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.config.milestone_reward.enable and self.config.milestone_reward.strategy == "hit_only":
                            milestone_rewards = []
                            similarity_matches = []
                            num_steps_in_batch = len(batch.non_tensor_batch["uid"])
                            # decay settings for repeated milestone hits within the same uid/task
                            decay_gamma = float(getattr(self.config.milestone_reward, "decay_gamma", 0.8))
                            decay_gamma = max(0.0, min(1.0, decay_gamma))
                            milestone_hit_counts = {}
                            milestone_progress = {}
                            for i in range(num_steps_in_batch):
                                task = batch.non_tensor_batch["task"][i]
                                action_desc = batch.non_tensor_batch["action_description"][i] 

                                if task not in self.milestone_dict:
                                    milestone_rewards.append(0)
                                    continue

                                if task not in self._sc_milestone_embed_cache:
                                    self._sc_milestone_embed_cache[task] = self.bert_embedder.embed(self.milestone_dict[task])
                                milestone_embeddings = self._sc_milestone_embed_cache[task]

                                act_embeddings = self.bert_embedder.embed([action_desc])

                                worker_id = batch.non_tensor_batch["worker_id"][i]
                                progress_key = (worker_id, task)
                                current_index = milestone_progress.get(progress_key, 0)

                                if current_index >= len(self.milestone_dict[task]):
                                    milestone_rewards.append(0)
                                    continue

                                cos_sim = torch.nn.functional.cosine_similarity(act_embeddings, milestone_embeddings, dim=-1)
                                current_cos_sim = float(cos_sim[current_index])

                                if current_cos_sim > self.config.milestone_reward.threshold:
                                    uid = batch.non_tensor_batch["uid"][i]
                                    key = (uid, task, current_index)
                                    prev_count = milestone_hit_counts.get(key, 0)
                                    occurrence_index = prev_count + 1
                                    milestone_hit_counts[key] = occurrence_index
                                    decayed_reward = current_cos_sim * (decay_gamma ** (occurrence_index - 1))
                                    milestone_rewards.append(decayed_reward)
                                    milestone_sentence = self.milestone_dict[task][current_index] if task in self.milestone_dict and current_index < len(self.milestone_dict[task]) else ""
                                    similarity_matches.append(
                                        {
                                            "task": task,
                                            "milestone_sentence": milestone_sentence,
                                            "action_sentence": action_desc,
                                            "cosine_similarity": current_cos_sim,
                                        }
                                    )
                                    milestone_progress[progress_key] = current_index + 1
                                    if (task, current_index) in self.milestone_history_hit_counts:
                                        self.milestone_history_hit_counts[(task, current_index)]["hit_count"] += 1
                                    else:
                                        self.milestone_history_hit_counts[(task, current_index)] = {"hit_count": 1, "threshold": self.config.milestone_reward.threshold}
                                else:
                                    milestone_rewards.append(0)
                            batch.batch["milestone_rewards"] = torch.tensor([float(x) for x in milestone_rewards], dtype=torch.float32)
                            if similarity_matches:
                                try:
                                    if os.path.exists(self.milestone_similarity_log_path):
                                        with open(self.milestone_similarity_log_path, "r") as f:
                                            existing_content = json.load(f)
                                    else:
                                        existing_content = {"matches": []}
                                    if not isinstance(existing_content, dict):
                                        existing_content = {"matches": []}
                                    existing_content.setdefault("matches", [])
                                    existing_content.setdefault("stats", {"total": 0})
                                    existing_content["matches"].extend(similarity_matches)
                                    existing_content["stats"]["total"] = existing_content["stats"].get("total", 0) + len(similarity_matches)
                                    with open(self.milestone_similarity_log_path, "w") as f:
                                        json.dump(existing_content, f, indent=2, ensure_ascii=False)
                                except Exception as e:
                                    pprint(f"Failed to log milestone similarity matches: {e}")
                        
                        elif self.config.milestone_reward.enable and self.config.milestone_reward.strategy == "previous":
                            num_steps_in_batch = len(batch.non_tensor_batch["uid"])
                            milestone_rewards = [0.0 for _ in range(num_steps_in_batch)]
                            similarity_matches = []
                            milestone_progress = {}
                            worker_task_indices = {}
                            for i in range(num_steps_in_batch):
                                task = batch.non_tensor_batch["task"][i]
                                action_desc = batch.non_tensor_batch["action_description"][i] 

                                if task not in self.milestone_dict:
                                    continue

                                if task not in self._sc_milestone_embed_cache:
                                    self._sc_milestone_embed_cache[task] = self.bert_embedder.embed(self.milestone_dict[task])
                                milestone_embeddings = self._sc_milestone_embed_cache[task]

                                act_embeddings = self.bert_embedder.embed([action_desc])

                                worker_id = batch.non_tensor_batch["worker_id"][i]
                                progress_key = (worker_id, task)
                                step_indices = worker_task_indices.setdefault(progress_key, [])
                                step_indices.append(i)

                                total_milestones = len(self.milestone_dict[task])
                                if total_milestones <= 0:
                                    continue

                                current_hits = milestone_progress.get(progress_key, 0)
                                current_fraction = current_hits / total_milestones
                                milestone_rewards[i] = current_fraction

                                if current_hits >= total_milestones:
                                    continue

                                cos_sim = torch.nn.functional.cosine_similarity(act_embeddings, milestone_embeddings, dim=-1)
                                current_cos_sim = float(cos_sim[current_hits])

                                if current_cos_sim > self.config.milestone_reward.threshold:
                                    new_hits = current_hits + 1
                                    new_fraction = new_hits / total_milestones
                                    milestone_sentence = self.milestone_dict[task][current_hits] if current_hits < len(self.milestone_dict[task]) else ""

                                    for idx in step_indices:
                                        milestone_rewards[idx] = new_fraction

                                    milestone_progress[progress_key] = new_hits
                                    similarity_matches.append(
                                        {
                                            "task": task,
                                            "milestone_sentence": milestone_sentence,
                                            "action_sentence": action_desc,
                                            "cosine_similarity": current_cos_sim,
                                        }
                                    )

                                    if (task, current_hits) in self.milestone_history_hit_counts:
                                        self.milestone_history_hit_counts[(task, current_hits)]["hit_count"] += 1
                                    else:
                                        self.milestone_history_hit_counts[(task, current_hits)] = {"hit_count": 1, "threshold": self.config.milestone_reward.threshold}
                            batch.batch["milestone_rewards"] = torch.tensor([float(x) for x in milestone_rewards], dtype=torch.float32)
                            if similarity_matches:
                                try:
                                    if os.path.exists(self.milestone_similarity_log_path):
                                        with open(self.milestone_similarity_log_path, "r") as f:
                                            existing_content = json.load(f)
                                    else:
                                        existing_content = {"matches": []}
                                    if not isinstance(existing_content, dict):
                                        existing_content = {"matches": []}
                                    existing_content.setdefault("matches", [])
                                    existing_content.setdefault("stats", {"total": 0})
                                    existing_content["matches"].extend(similarity_matches)
                                    existing_content["stats"]["total"] = existing_content["stats"].get("total", 0) + len(similarity_matches)
                                    with open(self.milestone_similarity_log_path, "w") as f:
                                        json.dump(existing_content, f, indent=2, ensure_ascii=False)
                                except Exception as e:
                                    pprint(f"Failed to log milestone similarity matches: {e}")
                        
                        elif self.config.milestone_reward.enable and self.config.milestone_reward.strategy == "mix":
                            num_steps_in_batch = len(batch.non_tensor_batch["uid"])
                            milestone_rewards = [0.0 for _ in range(num_steps_in_batch)]
                            similarity_matches = []
                            
                            
                            accuracy_rewards = batch.batch.get("accuracy_rewards", None)
                            if accuracy_rewards is None:
                               
                                accuracy_rewards = torch.zeros(num_steps_in_batch)
                            
                            # decay settings for hit_only part
                            decay_gamma = float(getattr(self.config.milestone_reward, "decay_gamma", 0.8))
                            decay_gamma = max(0.0, min(1.0, decay_gamma))
                            milestone_hit_counts = {}
                            
                            
                            milestone_progress = {}
                            
                            worker_task_indices = {}
                            
                            for i in range(num_steps_in_batch):
                                task = batch.non_tensor_batch["task"][i]
                                action_desc = batch.non_tensor_batch["action_description"][i]
                                is_correct = float(accuracy_rewards[i]) == 1.0
                                
                                if task not in self.milestone_dict:
                                    
                                    continue
                                
                                if task not in self._sc_milestone_embed_cache:
                                    self._sc_milestone_embed_cache[task] = self.bert_embedder.embed(self.milestone_dict[task])
                                milestone_embeddings = self._sc_milestone_embed_cache[task]
                                
                                act_embeddings = self.bert_embedder.embed([action_desc])
                                
                                worker_id = batch.non_tensor_batch["worker_id"][i]
                                progress_key = (worker_id, task)
                                
                                if is_correct:
                                    
                                    current_index = milestone_progress.get(progress_key, 0)
                                    
                                    if current_index >= len(self.milestone_dict[task]):
                                        
                                        continue
                                    
                                    cos_sim = torch.nn.functional.cosine_similarity(act_embeddings, milestone_embeddings, dim=-1)
                                    current_cos_sim = float(cos_sim[current_index])
                                    
                                    if current_cos_sim > self.config.milestone_reward.threshold:
                                        
                                        uid = batch.non_tensor_batch["uid"][i]
                                        key = (uid, task, current_index)
                                        prev_count = milestone_hit_counts.get(key, 0)
                                        occurrence_index = prev_count + 1
                                        milestone_hit_counts[key] = occurrence_index
                                        decayed_reward = current_cos_sim * (decay_gamma ** (occurrence_index - 1))
                                        milestone_rewards[i] = decayed_reward
                                        
                                        milestone_sentence = self.milestone_dict[task][current_index] if current_index < len(self.milestone_dict[task]) else ""
                                        similarity_matches.append(
                                            {
                                                "task": task,
                                                "milestone_sentence": milestone_sentence,
                                                "action_sentence": action_desc,
                                                "cosine_similarity": current_cos_sim,
                                                "trajectory_type": "correct",
                                            }
                                        )
                                        milestone_progress[progress_key] = current_index + 1
                                        
                                        if (task, current_index) in self.milestone_history_hit_counts:
                                            self.milestone_history_hit_counts[(task, current_index)]["hit_count"] += 1
                                        else:
                                            self.milestone_history_hit_counts[(task, current_index)] = {"hit_count": 1, "threshold": self.config.milestone_reward.threshold}
                                else:
                                    
                                    step_indices = worker_task_indices.setdefault(progress_key, [])
                                    step_indices.append(i)
                                    
                                    total_milestones = len(self.milestone_dict[task])
                                    if total_milestones <= 0:
                                        continue
                                    
                                    current_hits = milestone_progress.get(progress_key, 0)
                                    
                                    current_fraction = current_hits / total_milestones
                                    milestone_rewards[i] = current_fraction
                                    
                                    if current_hits >= total_milestones:
                                        
                                        continue
                                    
                                    cos_sim = torch.nn.functional.cosine_similarity(act_embeddings, milestone_embeddings, dim=-1)
                                    current_cos_sim = float(cos_sim[current_hits])
                                    
                                    if current_cos_sim > self.config.milestone_reward.threshold:
                                        
                                        uid = batch.non_tensor_batch["uid"][i]
                                        key = (uid, task, current_hits)
                                        prev_count = milestone_hit_counts.get(key, 0)
                                        occurrence_index = prev_count + 1
                                        milestone_hit_counts[key] = occurrence_index
                                        
                                        
                                        hit_bonus = current_cos_sim * (decay_gamma ** (occurrence_index - 1))
                                        new_hits = current_hits + 1
                                        new_fraction = new_hits / total_milestones
                                        combined_reward = new_fraction + hit_bonus*0.3
                                        
                                        
                                        milestone_rewards[i] = combined_reward
                                        
                                        old_fraction = current_fraction
                                        for idx in step_indices[:-1]:  
                                            if milestone_rewards[idx] == old_fraction:
                                                milestone_rewards[idx] = new_fraction
                                            else:
                                                milestone_rewards[idx] = new_fraction + milestone_rewards[idx]-old_fraction
                                        
                                        milestone_sentence = self.milestone_dict[task][current_hits] if current_hits < len(self.milestone_dict[task]) else ""
                                        similarity_matches.append(
                                            {
                                                "task": task,
                                                "milestone_sentence": milestone_sentence,
                                                "action_sentence": action_desc,
                                                "cosine_similarity": current_cos_sim,
                                                "trajectory_type": "incorrect",
                                                "hit_bonus": hit_bonus,
                                            }
                                        )
                                        
                                        milestone_progress[progress_key] = new_hits
                                        
                                        if (task, current_hits) in self.milestone_history_hit_counts:
                                            self.milestone_history_hit_counts[(task, current_hits)]["hit_count"] += 1
                                        else:
                                            self.milestone_history_hit_counts[(task, current_hits)] = {"hit_count": 1, "threshold": self.config.milestone_reward.threshold}
                            
                            batch.batch["milestone_rewards"] = torch.tensor([float(x) for x in milestone_rewards], dtype=torch.float32)
                            if similarity_matches:
                                try:
                                    if os.path.exists(self.milestone_similarity_log_path):
                                        with open(self.milestone_similarity_log_path, "r") as f:
                                            existing_content = json.load(f)
                                    else:
                                        existing_content = {"matches": []}
                                    if not isinstance(existing_content, dict):
                                        existing_content = {"matches": []}
                                    existing_content.setdefault("matches", [])
                                    existing_content.setdefault("stats", {"total": 0})
                                    existing_content["matches"].extend(similarity_matches)
                                    existing_content["stats"]["total"] = existing_content["stats"].get("total", 0) + len(similarity_matches)
                                    with open(self.milestone_similarity_log_path, "w") as f:
                                        json.dump(existing_content, f, indent=2, ensure_ascii=False)
                                except Exception as e:
                                    pprint(f"Failed to log milestone similarity matches: {e}")
                        
                        elif self.config.milestone_reward.enable and self.config.milestone_reward.strategy == "model_judge_mix":
                            # Model-based judgment version of mix strategy
                            # Uses binary rewards (1 or 0) from model judgment instead of cosine similarity
                            num_steps_in_batch = len(batch.non_tensor_batch["uid"])
                            milestone_rewards = [0.0 for _ in range(num_steps_in_batch)]
                            similarity_matches = []
                            
                            accuracy_rewards = batch.batch.get("accuracy_rewards", None)
                            if accuracy_rewards is None:
                                accuracy_rewards = torch.zeros(num_steps_in_batch)
                            
                            # decay settings for hit_only part
                            decay_gamma = float(getattr(self.config.milestone_reward, "decay_gamma", 0.8))
                            decay_gamma = max(0.0, min(1.0, decay_gamma))
                            milestone_hit_counts = {}
                            
                            milestone_progress = {}
                            worker_task_indices = {}
                            
                            for i in range(num_steps_in_batch):
                                task = batch.non_tensor_batch["task"][i]
                                action_desc = batch.non_tensor_batch["action_description"][i]
                                is_correct = float(accuracy_rewards[i]) == 1.0
                                
                                if task not in self.milestone_dict:
                                    continue
                                
                                worker_id = batch.non_tensor_batch["worker_id"][i]
                                progress_key = (worker_id, task)
                                
                                if is_correct:
                                    # Correct trajectory: use hit_only strategy with model judgment
                                    current_index = milestone_progress.get(progress_key, 0)
                                    
                                    if current_index >= len(self.milestone_dict[task]):
                                        continue
                                    
                                    # Use model to judge if action matches milestone (returns 1 or 0)
                                    milestone_sentence = self.milestone_dict[task][current_index]
                                    try:
                                        match_score = self.milestone_judge.judge_pair(milestone_sentence, action_desc)
                                    except Exception as exc:
                                        logger.error(f"MilestoneJudge call failed: {exc}")
                                        match_score = 0
                                    
                                    if match_score == 1:  # Model says it matches
                                        uid = batch.non_tensor_batch["uid"][i]
                                        key = (uid, task, current_index)
                                        prev_count = milestone_hit_counts.get(key, 0)
                                        occurrence_index = prev_count + 1
                                        milestone_hit_counts[key] = occurrence_index
                                        # Use binary reward with decay
                                        decayed_reward = 1.0 * (decay_gamma ** (occurrence_index - 1))
                                        milestone_rewards[i] = decayed_reward
                                        
                                        similarity_matches.append(
                                            {
                                                "task": task,
                                                "milestone_sentence": milestone_sentence,
                                                "action_sentence": action_desc,
                                                "match_score": match_score,
                                                "trajectory_type": "correct",
                                            }
                                        )
                                        milestone_progress[progress_key] = current_index + 1
                                        
                                        if (task, current_index) in self.milestone_history_hit_counts:
                                            self.milestone_history_hit_counts[(task, current_index)]["hit_count"] += 1
                                        else:
                                            self.milestone_history_hit_counts[(task, current_index)] = {"hit_count": 1, "threshold": "model_judge"}
                                else:
                                    # Incorrect trajectory: use previous strategy with model judgment
                                    step_indices = worker_task_indices.setdefault(progress_key, [])
                                    step_indices.append(i)
                                    
                                    total_milestones = len(self.milestone_dict[task])
                                    if total_milestones <= 0:
                                        continue
                                    
                                    current_hits = milestone_progress.get(progress_key, 0)
                                    
                                    current_fraction = current_hits / total_milestones
                                    milestone_rewards[i] = current_fraction
                                    
                                    if current_hits >= total_milestones:
                                        continue
                                    
                                    # Use model to judge if action matches milestone
                                    milestone_sentence = self.milestone_dict[task][current_hits]
                                    try:
                                        match_score = self.milestone_judge.judge_pair(milestone_sentence, action_desc)
                                    except Exception as exc:
                                        logger.error(f"MilestoneJudge call failed: {exc}")
                                        match_score = 0
                                    
                                    if match_score == 1:  # Model says it matches
                                        uid = batch.non_tensor_batch["uid"][i]
                                        key = (uid, task, current_hits)
                                        prev_count = milestone_hit_counts.get(key, 0)
                                        occurrence_index = prev_count + 1
                                        milestone_hit_counts[key] = occurrence_index
                                        
                                        # Binary hit bonus with decay
                                        hit_bonus = 1.0 * (decay_gamma ** (occurrence_index - 1))
                                        new_hits = current_hits + 1
                                        new_fraction = new_hits / total_milestones
                                        combined_reward = new_fraction + hit_bonus * 0.3
                                        
                                        milestone_rewards[i] = combined_reward
                                        
                                        old_fraction = current_fraction
                                        for idx in step_indices[:-1]:
                                            if milestone_rewards[idx] == old_fraction:
                                                milestone_rewards[idx] = new_fraction
                                            else:
                                                milestone_rewards[idx] = new_fraction + milestone_rewards[idx] - old_fraction
                                        
                                        similarity_matches.append(
                                            {
                                                "task": task,
                                                "milestone_sentence": milestone_sentence,
                                                "action_sentence": action_desc,
                                                "match_score": match_score,
                                                "trajectory_type": "incorrect",
                                                "hit_bonus": hit_bonus,
                                            }
                                        )
                                        
                                        milestone_progress[progress_key] = new_hits
                                        
                                        if (task, current_hits) in self.milestone_history_hit_counts:
                                            self.milestone_history_hit_counts[(task, current_hits)]["hit_count"] += 1
                                        else:
                                            self.milestone_history_hit_counts[(task, current_hits)] = {"hit_count": 1, "threshold": "model_judge"}
                            
                            batch.batch["milestone_rewards"] = torch.tensor([float(x) for x in milestone_rewards], dtype=torch.float32)
                            if similarity_matches:
                                try:
                                    if os.path.exists(self.milestone_similarity_log_path):
                                        with open(self.milestone_similarity_log_path, "r") as f:
                                            existing_content = json.load(f)
                                    else:
                                        existing_content = {"matches": []}
                                    if not isinstance(existing_content, dict):
                                        existing_content = {"matches": []}
                                    existing_content.setdefault("matches", [])
                                    existing_content.setdefault("stats", {"total": 0})
                                    existing_content["matches"].extend(similarity_matches)
                                    existing_content["stats"]["total"] = existing_content["stats"].get("total", 0) + len(similarity_matches)
                                    with open(self.milestone_similarity_log_path, "w") as f:
                                        json.dump(existing_content, f, indent=2, ensure_ascii=False)
                                except Exception as e:
                                    pprint(f"Failed to log milestone similarity matches: {e}")


                        # ---- Step Helpfulness Reward Calculation ----
                        if self.config.step_judge.enable:
                            num_steps_in_batch = len(batch.non_tensor_batch["uid"])

                            def _judge_helpfulness(i):
                                goal = batch.non_tensor_batch["goal"][i]
                                action_text = batch.non_tensor_batch["action_text"][i]
                                image_path = batch.non_tensor_batch["image_path"][i]
                                action_desc = batch.non_tensor_batch["action_description"][i]
                                task = batch.non_tensor_batch["task"][i]
                                if task not in self.step_judge.ref_task_dict:
                                    judge_result = self.step_judge.judge_step(
                                        goal=goal,
                                        action_text=action_text,
                                        image_path=image_path,
                                        action_description=action_desc,
                                    )
                                else:
                                    judge_result = self.step_judge.judge_step_with_reference(
                                        goal=goal,
                                        action_text=action_text,
                                        image_path=image_path,
                                        action_description=action_desc,
                                        task=task,
                                    )
                                return 1.0 if judge_result.get("label") == "yes" else 0.0

                            max_workers = min(num_steps_in_batch, 16)
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                helpfulness_rewards = list(tqdm(executor.map(_judge_helpfulness, range(num_steps_in_batch)), total=num_steps_in_batch, desc="Judging Steps"))

                            batch.batch["helpfulness_rewards"] = torch.tensor([float(x) for x in helpfulness_rewards], dtype=torch.float32)
                            
                        

                        


                        if not self.config.strategy.use_validity_reward:
                            rewards = (
                                batch.batch["accuracy_rewards"] + 0.5 * batch.batch["format_rewards"] + self.config.strategy.punish_coef * batch.batch["punishment"]
                            )
                        else:
                            rewards = (
                                torch.where(
                                    batch.batch["accuracy_rewards"] == 1, 
                                    batch.batch["accuracy_rewards"] + batch.batch["validity_rewards"], 
                                    batch.batch["accuracy_rewards"]
                                ) + 0.5 * batch.batch["format_rewards"] + self.config.strategy.punish_coef * batch.batch["punishment"]
                            )
                        # process rewards is all zero with the same shape as rewards
                        process_rewards = torch.zeros_like(rewards)
                        if "confidence" in batch.non_tensor_batch:
                            confidence_rewards = torch.tensor([(float(C)-float(A))*(float(C)-float(A)) for C,A in zip(batch.non_tensor_batch["confidence"],batch.batch["accuracy_rewards"])], dtype=torch.float32)
                            batch.batch["confidence_rewards"]=confidence_rewards
                            # 把batch.batch["accuracy_rewards"]为1的对应项的confidence_rewards设置为0
                            batch.batch["confidence_rewards"] = torch.where(batch.batch["accuracy_rewards"] == 1, torch.zeros_like(batch.batch["confidence_rewards"]), batch.batch["confidence_rewards"])
                        


                        if self.config.milestone_reward.enable and "milestone_rewards" in batch.batch:
                            process_rewards += batch.batch["milestone_rewards"]*self.config.milestone_reward.weight

                        if self.config.img_milestone_reward.enable and "img_milestone_rewards" in batch.batch:
                            process_rewards += batch.batch["img_milestone_rewards"] * self.config.img_milestone_reward.weight
                                                
                        # Integrate the helpfulness reward
                        if self.step_judge and "helpfulness_rewards" in batch.batch:
                            # If judge is enabled, use its reward as the primary signal
                            process_rewards += batch.batch["helpfulness_rewards"]* self.config.step_judge.weight

                        if self.config.confidence_reward.enable:
                            process_rewards=-self.config.confidence_reward.weight*confidence_rewards
                        
                        if self.config.step_milestone.enable and "SC_rewards" in batch.batch:
                            process_rewards += batch.batch["SC_rewards"] * self.config.step_milestone.weight

                        if self.config.process_reward.enable:
                            process_rewards=process_rewards*self.config.process_reward.weight
                            if self.config.process_reward.decay_gamma is not None:
                                process_rewards=process_rewards*self.config.process_reward.decay_gamma**(epoch)
                            rewards=rewards+process_rewards

                        batch.batch["rewards"] = rewards

                        uid_set = set(batch.non_tensor_batch["uid"])

                        for uid in uid_set:
                            worker_set = set(batch.non_tensor_batch["worker_id"][batch.non_tensor_batch["uid"] == uid])
                            # breakpoint()
                            #计算success worker的数量
                            cond1 = torch.from_numpy(batch.non_tensor_batch["uid"] == uid)  # 转换为 tensor
                            cond2 = batch.batch["accuracy_rewards"] == 1    
                            success_worker_batch = batch.non_tensor_batch["worker_id"][(cond1 & cond2).numpy()]
                            success_worker_set = set(success_worker_batch)
                            
                            cur_difficult= 1-len(success_worker_set)/len(worker_set)
                            # breakpoint()
                            cur_task = batch.non_tensor_batch["task"][batch.non_tensor_batch["uid"] == uid].tolist()[0]
                            if cur_task not in self.task_difficulty_dict:
                                self.task_difficulty_dict[cur_task] = cur_difficult
                            else:
                                self.task_difficulty_dict[cur_task] = 0.5*self.task_difficulty_dict[cur_task] + 0.5*cur_difficult
                            
                            mean_accuracy_reward = batch.batch["accuracy_rewards"][batch.non_tensor_batch["uid"] == uid].mean().item()
                            mean_format_reward = batch.batch["format_rewards"][batch.non_tensor_batch["uid"] == uid].mean().item()
                            mean_accuracy_reward_with_punish = (batch.batch["accuracy_rewards"] + self.config.strategy.punish_coef * batch.batch["punishment"])[batch.non_tensor_batch["uid"] == uid].mean().item()
                            self.mean_accuracy_reward_map[uid] = mean_accuracy_reward
                            if not self.config.strategy.use_validity_reward:
                                pprint(f"uid: {uid}, accuracy reward: {mean_accuracy_reward}, format reward: {mean_format_reward}, accuracy reward with punish: {mean_accuracy_reward_with_punish}")
                            else:
                                mean_validity_reward = batch.batch["validity_rewards"][batch.non_tensor_batch["uid"] == uid].mean().item()
                                pprint(f"uid: {uid}, accuracy reward: {mean_accuracy_reward}, format reward: {mean_format_reward}, accuracy reward with punish: {mean_accuracy_reward_with_punish}, validity_reward: {mean_validity_reward}")
                            with open(f"{self.config.trainer.rollout_data_dir}/reward_logs.jsonl", 'a', encoding="utf-8") as outfile:
                                outfile.write(json.dumps({"uid": uid, "accuracy_reward": mean_accuracy_reward, "format_reward": mean_format_reward, "accuracy_reward_with_punish": mean_accuracy_reward_with_punish}, ensure_ascii=False) + "\n")

                        
                        if self.config.Certainty_Reward.enable:
                            for uid in uid_set:
                                cur_task = batch.non_tensor_batch["task"][batch.non_tensor_batch["uid"] == uid].tolist()[0]
                                if cur_task not in self.task_difficulty_dict:
                                    continue
                                if self.task_difficulty_dict[cur_task] < self.config.Certainty_Reward.threshold:
                                    batch.batch["rewards"][batch.non_tensor_batch["uid"] == uid] = batch.batch["rewards"][batch.non_tensor_batch["uid"] == uid] + self.config.Certainty_Reward.weight*batch.batch["confidence_rewards"][batch.non_tensor_batch["uid"] == uid]
                                else:
                                    batch.batch["rewards"][batch.non_tensor_batch["uid"] == uid] = batch.batch["rewards"][batch.non_tensor_batch["uid"] == uid] - self.config.Certainty_Reward.weight*batch.batch["confidence_rewards"][batch.non_tensor_batch["uid"] == uid]

          
                        reward_stds_list = []
                        for uid in uid_set:
                            reward_in_group = batch.batch["rewards"][batch.non_tensor_batch["uid"] == uid]
                            # compute std in group
                            reward_std = reward_in_group.std().item()
                            if np.isnan(reward_std):
                                reward_std = 0
                                reward_in_group *= 0
                                pprint(reward_in_group)
                                pprint(f"reward_in_group shape: {reward_in_group.shape}")
                            reward_stds_list.append(reward_std)

                        num_invalid_group = len(
                            [x_std for x_std in reward_stds_list if x_std < 0.01]
                        )
                        pprint(
                            f"num_invalid_group: {num_invalid_group}/{len(reward_stds_list)} | reward_stds_list: {reward_stds_list}"
                        )
                        if self.config.strategy.skip_invalid_groups:
                            if num_invalid_group != 0:
                                pprint(f"num_invalid_group != 0, skip this batch !!!")
                                continue

                        # we combine with rule-based rm
                        reward_tensor = batch.batch["rewards"]
                        reward_extra_infos_dict = {"uid": batch.non_tensor_batch["uid"], "worker_id": batch.non_tensor_batch["worker_id"]}
                        reward_metrics = {
                            "reward_tensor": reward_tensor.tolist(),
                            "reward_std": reward_stds_list,
                            "accuracy_rewards": batch.batch["accuracy_rewards"].tolist(),
                            "format_rewards": batch.batch["format_rewards"].tolist(),
                            "confidence": batch.non_tensor_batch["confidence"].tolist() if "confidence" in batch.non_tensor_batch else None,
                            "kl_uniform": batch.non_tensor_batch["kl_uniform"].tolist() if "kl_uniform" in batch.non_tensor_batch else None,
                            "confidence_rewards":batch.batch["confidence_rewards"].tolist() if "confidence_rewards" in batch.batch else None,
                            "num_invalid_group": num_invalid_group,
                        }
                        if self.config.strategy.use_validity_reward:
                            reward_metrics["validity_rewards"] = batch.batch["validity_rewards"].tolist()
                        if self.config.strategy.use_replay:
                            reward_metrics["punishment"] = batch.batch["punishment"].tolist()
                        
                        if self.step_judge and "helpfulness_rewards" in batch.batch:
                            reward_metrics["helpfulness_rewards"] = batch.batch["helpfulness_rewards"].tolist()
                        if self.config.milestone_reward.enable and "milestone_rewards" in batch.batch:
                            reward_metrics["milestone_rewards"] = batch.batch["milestone_rewards"].tolist()
                        if self.config.img_milestone_reward.enable and "img_milestone_rewards" in batch.batch:
                            reward_metrics["img_milestone_rewards"] = batch.batch["img_milestone_rewards"].tolist()
                        if self.config.step_milestone.enable and "SC_rewards" in batch.batch:
                            reward_metrics["SC_rewards"] = batch.batch["SC_rewards"].tolist()

                        batch.batch["token_level_scores"] = reward_tensor.unsqueeze(-1)
                        reward_metrics = {
                            f"reward/{key}": value
                            for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)




                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        batch_padded, pad_size = pad_dataproto_to_constant_size(
                            batch, 
                            self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n,
                        )
                        pprint(f"batch_padded len: {len(batch_padded)}")
                        old_log_prob_padded = self.actor_rollout_wg.compute_log_prob(batch_padded)
                        old_log_prob = unpad_dataproto(old_log_prob_padded, pad_size=pad_size)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            batch_padded, pad_size = pad_dataproto_to_constant_size(
                                batch, 
                                self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n,
                            )
                            if not self.ref_in_actor:
                                ref_log_prob_padded = self.ref_policy_wg.compute_ref_log_prob(batch_padded)
                            else:
                                ref_log_prob_padded = self.actor_rollout_wg.compute_ref_log_prob(batch_padded)
                            ref_log_prob = unpad_dataproto(ref_log_prob_padded, pad_size=pad_size)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        # reward_extra_infos_dict: dict[str, list]
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm
                        )

                        # Apply difficulty factor to advantages
                        if self.config.Difficulty_factor.enable:
                            for uid in uid_set:
                                llama_diff=0.8
                                cur_task = batch.non_tensor_batch["task"][batch.non_tensor_batch["uid"] == uid].tolist()[0]
                                mul_factor = llama_diff+(1-llama_diff)*self.task_difficulty_dict[cur_task]
                                batch.batch["advantages"][batch.non_tensor_batch["uid"] == uid] = batch.batch["advantages"][batch.non_tensor_batch["uid"] == uid] * mul_factor

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            batch_padded, pad_size = pad_dataproto_to_constant_size(
                                batch, 
                                self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
                            )
                            batch_padded.batch["advantages"][-pad_size:,] = 0
                            actor_output_padded = self.actor_rollout_wg.update_actor(batch_padded)
                            actor_output = unpad_dataproto(actor_output_padded, pad_size=pad_size)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # # validate_static
                    # with _timer("testing_static", timing_raw):
                    #     static_val_metrics: dict = self._validate_static()
                    #     metrics.update(static_val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        
                        # Save milestone data and task difficulty at the same frequency as checkpoint saving
                        try:
                            # Convert tuple keys to string keys for JSON serialization
                            serializable_data = {}
                            for key, value in self.milestone_history_hit_counts.items():
                                if isinstance(key, tuple):
                                    serializable_data[f"{key[0]}_{key[1]}"] = value
                                else:
                                    serializable_data[str(key)] = value
                            
                            with open(self.milestone_history_hit_counts_path+f"/milestone_history_hit_counts_step_{self.global_steps}.json", 'w') as f:
                                json.dump(serializable_data, f, indent=2)
                            pprint(f"Saved milestone_history_hit_counts for step {self.global_steps}")
                        except Exception as e:
                            pprint(f"Error saving milestone_history_hit_counts for step {self.global_steps}: {e}")
                        
                        # Save updated milestone_dict
                        try:
                            with open(self.milestone_history_hit_counts_path+f"/milestone_dict_step_{self.global_steps}.json", 'w') as f:
                                json.dump(self.milestone_dict, f, indent=2, ensure_ascii=False)
                            pprint(f"Saved milestone_dict for step {self.global_steps}")
                        except Exception as e:
                            pprint(f"Error saving milestone_dict for step {self.global_steps}: {e}")
                        
                        # Save task difficulty dict
                        try:
                            with open(self.milestone_history_hit_counts_path+f"/task_difficulty_dict_step_{self.global_steps}.json", 'w') as f:
                                json.dump(self.task_difficulty_dict, f, indent=2, ensure_ascii=False)
                            pprint(f"Saved task_difficulty_dict for step {self.global_steps}")
                        except Exception as e:
                            pprint(f"Error saving task_difficulty_dict for step {self.global_steps}: {e}")

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

        self._close_envs()