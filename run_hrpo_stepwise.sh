#!/bin/bash

# ============================================================
# ADMIRE Training Script
# Please install dependencies first: see README.md
# ============================================================

set -x

# Check Ray status
if ! command -v ray &> /dev/null; then
    echo "Ray is not installed. Please run: pip install ray"
    exit 1
fi

if ray status &> /dev/null; then
    echo "Ray is running"
    DASH_URL=$(ray status | grep "Dashboard" | awk '{print $2}')
    echo "Dashboard: $DASH_URL"
else
    echo "Ray is not running, starting..."
    ray start --head --port=6379 --dashboard-port=8265
fi

# Project paths
ROOT_PATH="."
PROJECT_NAME="aw_online_rl"
EXPERIMENT_NAME="test"
DEFAULT_LOCAL_DIR="${ROOT_PATH}/saves/${PROJECT_NAME}/${EXPERIMENT_NAME}" && mkdir -p "$DEFAULT_LOCAL_DIR"
DATETIME="$(date +%Y%m%d_%H%M%S)"
VALIDATION_DATA_DIR="${DEFAULT_LOCAL_DIR}/${DATETIME}/valid" && mkdir -p "$VALIDATION_DATA_DIR"
ROLLOUT_DATA_DIR="${DEFAULT_LOCAL_DIR}/${DATETIME}/rollout" && mkdir -p "$ROLLOUT_DATA_DIR"

# Data paths
TRAIN_DATA_PATH="train_noa11y.json"
VAL_DATA_PATH="val.json"

# Model path
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# Replay buffer path
REPLAY_SAVE_PATH="${ROOT_PATH}/saves/replay" && mkdir -p "$REPLAY_SAVE_PATH"
REPLAY_SAVE_PATH="${REPLAY_SAVE_PATH}/${EXPERIMENT_NAME}-replay-${DATETIME}.pkl"

# Environment server URL (modify as needed)
# url="http://localhost:7860/"
# python -c "from src.hammer_server.client import HammerEnvClient; client = HammerEnvClient(\"${url}\"); client.release_all_devices()"

export HYDRA_FULL_ERROR=1
python -m hammer_trainer_stepwise.main_ppo \
    --config-path=${ROOT_PATH}/scripts \
    --config-name=config_stepwise_32.yaml \
    env.max_envs=[16] \
    env.src=[""] \
    env.max_steps=30 \
    env.val_max_steps=40 \
    env.train_max_tries_per_rollout=1 \
    env.val_max_tries_per_rollout=3 \
    env.max_history_descs=null \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    data.train_batch_size=4 \
    data.truncation=left \
    data.shuffle=false \
    data.max_prompt_length=6500 \
    data.max_response_length=512 \
    actor_rollout_ref.rollout.max_model_len=7592 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=8 \
    strategy.downsample_rollout_n=8 \
    strategy.skip_invalid_groups=false \
    strategy.use_human_helps=false \
    strategy.use_replay=true \
    strategy.replay_save_path=${REPLAY_SAVE_PATH} \
    strategy.replay_load_path=null \
    strategy.trainer="hrpo_ray_trainer" \
    strategy.punish_coef=0.1 \
    strategy.replay_step_nums_tolerance=0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.shuffle=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=true \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=grpo \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.balance_batch=false \
    trainer.total_epochs=20 \
    trainer.total_training_steps=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.log_val_generations=0 \
    trainer.val_before_train=false \
    trainer.logger="[console,swanlab]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
    trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    ray_init.debug=true \
    milestone_reward.enable=true \
    milestone_reward.threshold=0.75 \
    milestone_reward.weight=0.3 \
    milestone_reward.load_path=null \
    milestone_reward.hit_num_load_path=null \
    milestone_reward.strategy="mix" \
    step_judge.enable=false \
    step_judge.address="" \
    step_judge.model="gpt-4o" \
    step_judge.api_key="" \
    step_judge.weight=0.3 \
    confidence_reward.enable=false \
    confidence_reward.weight=0.3 \
    Difficulty_factor.enable=false \
    Difficulty_factor.load_path=null \
    process_reward.enable=true \
    process_reward.weight=1.0 \
    process_reward.decay_gamma=0.99 \
    "$@"
