# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Optional
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.metric_utils import (
    # compute_data_metrics,
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
    _timer,
)

import numpy as np
import ray
import torch
import uuid

from hammer_trainer.env import EnvWorker
from hammer_trainer.metric_utils import compute_data_metrics
from hammer_trainer.utils.dataset.rl_dataset import MessagesDataset, collate_fn_dummy

INIT_TASK_TIMEOUT = 10 * 60


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

    def _create_envs(self):
        max_envs = self.config.env.max_envs
        train_batch_size = self.config.data.train_batch_size
        n = self.config.actor_rollout_ref.rollout.n
        num_envs = train_batch_size * n
        assert (
            max_envs >= num_envs
        ), f"train_batch_size: {train_batch_size}, rollout_n: {n}, # of envs: {num_envs}, max_envs: {max_envs}"
        worker_id = f"env_worker_{uuid.uuid4().hex}"
        self.env_workers = [
            EnvWorker.remote(worker_id=worker_id, config=self.config) for _ in range(max_envs)
        ]

    def _get_alive_env_works(self):
        alives = ray.get([worker.is_alive.remote() for worker in self.env_workers])
        pprint(f"# of alive environments: {sum(alives)}")
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

    def _init_envs(self, batch_dict, is_training: bool = True):
        n = (
            self.config.actor_rollout_ref.rollout.n
            if is_training
            else self.config.actor_rollout_ref.rollout.val_kwargs.n
        )
        batch_task = [deepcopy(t) for t in batch_dict for _ in range(n)]
        num_env_workers = len(batch_task)
        alive_env_workers = self._get_alive_env_works()
        assert (
            len(alive_env_workers) >= num_env_workers
        ), f"# of alive env worker {len(alive_env_workers)} is not enough (< train_batch_size * rollout_n {len(batch_task)})."

        batch_env_workers = alive_env_workers[:num_env_workers]
        batch_env_output = ray.get(
            [worker.init_task.remote(task=t) for worker, t in zip(batch_env_workers, batch_task)],
            timeout=INIT_TASK_TIMEOUT,
        )
        return batch_task, batch_env_output, batch_env_workers

    def _reset_envs(self):
        ray.get([worker.reset.remote() for worker in self.env_workers])

    def _close_envs(self):
        ray.get([worker.release.remote() for worker in self.env_workers])

    def _generate_rollout_batch(self, batch_env_output):
        messages = [x["messages"] for x in batch_env_output]
        dataset = MessagesDataset(
            messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            fast_rollout=True,
        )

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(lambda x: dataset[x], range(len(dataset))))

        batch_dict = collate_fn(batch_dict)
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
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
        return gen_batch

    def _validate(self):
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            finished_test_batch_task = []
            finished_test_batch_env_workers = []
            batch_task, batch_env_output, batch_env_workers = self._init_envs(
                batch_dict=test_data, is_training=False
            )
            for _ in range(self.config.env.max_steps):
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

                gen_batch = self._generate_rollout_batch(batch_env_output=batch_env_output)
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
                batch_env_output = ray.get(
                    [
                        worker.step.remote(action_text)
                        for worker, action_text in zip(batch_env_workers, response_text_batch)
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
            sample_scores.extend(batch_scores)
            sample_inputs.extend(finished_test_batch_task + batch_task)
            batch_messages = ray.get(
                [
                    w.history_messages.remote()
                    for w in (finished_test_batch_env_workers + batch_env_workers)
                ]
            )
            sample_outputs.extend(batch_messages)

            reward_extra_infos_dict["reward"].extend(batch_scores)

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

        return {"val/reward_score": reward_score}

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
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:

                batch_size = len(batch_dict)

                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("init_envs", timing_raw):
                    batch_task, batch_env_output, batch_env_workers = self._init_envs(
                        batch_dict=batch_dict, is_training=True
                    )

                finished_train_batch_task = []
                finished_train_batch_env_output = []
                finished_train_batch_env_workers = []

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
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
                                gen_batch = self._generate_rollout_batch(
                                    batch_env_output=batch_env_output
                                )

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
                                gen_batch_padded_output = (
                                    self.async_rollout_manager.generate_sequences(gen_batch_padded)
                                )
                                self.async_rollout_manager.sleep()
                            # unpad
                            gen_batch_output = unpad_dataproto(
                                gen_batch_padded_output, pad_size=pad_size
                            )

                            response_text_batch = self.tokenizer.batch_decode(
                                gen_batch_output.batch["responses"],
                                skip_special_tokens=True,
                            )
                            with _timer("env_step", timing_raw):
                                batch_env_output = ray.get(
                                    [
                                        worker.step.remote(action_text)
                                        for worker, action_text in zip(
                                            batch_env_workers, response_text_batch
                                        )
                                    ]
                                )
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
                        num_total_samples = batch_size * self.config.actor_rollout_ref.rollout.n
                        num_failed_samples = (
                            num_total_samples - num_finished_samples - num_unfinished_samples
                        )

                        pprint(
                            f"# of total samples: {num_total_samples}, "
                            f"# of fininshed samples: {num_finished_samples}, "
                            f"# of unfinished samples: {num_unfinished_samples}, "
                            f"# of failed environment workers: {num_failed_samples}"
                        )
                        batch_task = finished_train_batch_task + batch_task
                        batch_env_output = finished_train_batch_env_output + batch_env_output
                        batch_env_workers = finished_train_batch_env_workers + batch_env_workers
                        # reward
                        # format reward
                        format_rewards = [out["format_reward"] for out in batch_env_output]
                        # accuracy reward
                        accuracy_rewards = ray.get(
                            [w.evaluate.remote() for w in finished_train_batch_env_workers]
                        )
                        pprint(f"accuracy_rewards: {accuracy_rewards}")
                        accuracy_rewards += [0] * num_unfinished_samples
                        assert len(format_rewards) == len(batch_task)
                        assert len(accuracy_rewards) == len(batch_task)

                    # grpo
                    with _timer("prepare_grpo_inputs", timing_raw):
                        batch_grpo_dict = ray.get(
                            [worker.input_tensors.remote() for worker in batch_env_workers]
                        )
                        batch = collate_fn(batch_grpo_dict)
                        batch = DataProto.from_single_dict(batch)

                        batch.batch["accuracy_rewards"] = torch.tensor(
                            [float(x) for x in accuracy_rewards], dtype=torch.float32
                        )
                        batch.batch["format_rewards"] = torch.tensor(
                            [float(x) for x in format_rewards], dtype=torch.float32
                        )
                        batch.non_tensor_batch["uid"] = np.array(
                            [x["task_id"] for x in batch_task], dtype=object
                        )

                    # reset environment workers
                    self._reset_envs()
                    # TODO replay

                    batch.batch["responses"] = batch.batch["input_ids"]
                    response_mask = batch.batch["labels"] != -100
                    batch.batch["response_mask"] = response_mask
                    batch.batch["loss_mask"] = response_mask[:, :-1]

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with _timer("reward", timing_raw):
                        rewards = (
                            batch.batch["accuracy_rewards"] + 0.5 * batch.batch["format_rewards"]
                        )
                        batch.batch["rewards"] = rewards

                        uid_set = set(batch.non_tensor_batch["uid"])

                        reward_stds_list = []
                        for uid in uid_set:
                            reward_in_group = batch.batch["rewards"][
                                batch.non_tensor_batch["uid"] == uid
                            ]
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

                        # we combine with rule-based rm
                        reward_tensor = batch.batch["rewards"]
                        reward_metrics = {
                            "reward_tensor": reward_tensor.tolist(),
                            "reward_std": reward_stds_list,
                            "num_invalid_group": num_invalid_group,
                            "accuracy_rewards": accuracy_rewards,
                            "foramt_rewards": format_rewards,
                        }

                        batch.batch["token_level_scores"] = reward_tensor.unsqueeze(-1)
                        reward_metrics = {
                            f"reward/{key}": value
                            for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)

                        if self.use_rm:
                            raise NotImplementedError("Reward model is not supported yet.")
                        # # compute reward model score
                        # if self.use_rm:
                        #     reward_tensor = self.rm_wg.compute_rm_score(batch)
                        #     batch = batch.union(reward_tensor)

                        # if self.config.reward_model.launch_reward_fn_async:
                        #     future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        # else:
                        #     reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        batch_padded, pad_size = pad_dataproto_to_divisor(
                            batch, self.actor_rollout_wg.world_size
                        )
                        old_log_prob_padded = self.actor_rollout_wg.compute_log_prob(batch_padded)
                        old_log_prob = unpad_dataproto(old_log_prob_padded, pad_size=pad_size)
                        # entropys = old_log_prob.batch["entropys"]
                        # response_masks = batch.batch["response_mask"]
                        # loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # entropy_loss = agg_loss(
                        #     loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        # )
                        # old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        # metrics.update(old_log_prob_metrics)
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
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool()
                            )
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
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # # we combine with rule-based rm
                        # reward_extra_infos_dict: dict[str, list]
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor

                        # if reward_extra_infos_dict:
                        #     batch.non_tensor_batch.update(
                        #         {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                        #     )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )
                        # batch.batch["advantages"] = batch.batch["advantages"][:, :-1]

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            batch_padded, pad_size = pad_dataproto_to_divisor(
                                batch,
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n,
                            )
                            critic_output_padded = self.critic_wg.update_critic(batch_padded)
                            critic_output = unpad_dataproto(critic_output_padded, pad_size=pad_size)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = True
                            batch_padded, pad_size = pad_dataproto_to_divisor(
                                batch,
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n,
                            )
                            advantages = batch_padded.batch["advantages"]
                            batch_padded.batch["advantages"] = advantages[:, :-1]
                            actor_output_padded = self.actor_rollout_wg.update_actor(batch_padded)
                            actor_output = unpad_dataproto(actor_output_padded, pad_size=pad_size)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # if rollout_data_dir:
                    #     with _timer("dump_rollout_generations", timing_raw):
                    #         print(batch.batch.keys())
                    #         inputs = self.tokenizer.batch_decode(
                    #             batch.batch["prompts"], skip_special_tokens=True
                    #         )
                    #         outputs = self.tokenizer.batch_decode(
                    #             batch.batch["responses"], skip_special_tokens=True
                    #         )
                    #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    #         self._dump_generations(
                    #             inputs=inputs,
                    #             outputs=outputs,
                    #             scores=scores,
                    #             reward_extra_infos_dict=reward_extra_infos_dict,
                    #             dump_path=rollout_data_dir,
                    #         )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

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
                metrics.update(
                    compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

        self._close_envs()
