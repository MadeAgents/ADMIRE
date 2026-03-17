# ADMIRE

<h1 align="center">Adaptive Milestone Reward for GUI Agents</h1>

<p align="center">
  <img src="framework.jpg" alt="ADMIRE Framework" width="100%">
</p>

<p align="center">
  <a href="README_zh.md">中文</a> | English
</p>

ADMIRE is a reinforcement learning framework that uses **adaptive milestone rewards** to train GUI agents. It automatically generates task milestones from successful trajectories and provides dense rewards to guide agent learning.

## Installation

### Requirements

- Python 3.11
- PyTorch 2.2.0
- CUDA 12.6
- 8× A800(A100) GPUs (recommended)

### Setup

```bash
# Create conda environment
conda create -n admire python=3.11 -y
conda activate admire

# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu126

# Clone repository
git clone https://github.com/your-repo/ADMIRE.git
cd ADMIRE

# Install verl
mkdir 3rdparty && git clone https://github.com/volcengine/verl 3rdparty/verl
cd 3rdparty/verl && pip install -e . && cd ../..

# Install android_world
git clone https://github.com/google-research/android_world 3rdparty/android_world
cd 3rdparty/android_world && pip install -e . && cd ../..

# Install ADMIRE
pip install -e .

# Install additional dependencies
pip install swanlab scikit-image spacy ray
python -m spacy download en_core_web_sm

# (Optional) Login to SwanLab for experiment tracking
swanlab login -k "your-api-key"
```

### Android Environment

Follow [AndroidWorld setup guide](3rdparty/android_world/README.md) to configure the Android emulator.

## Quick Start

### 1. Start Ray Cluster

```bash
ray stop
ray start --head --port=6379 --dashboard-port=8265
```

### 2. Start Environment Server

```bash
python src/hammer_server/gradio_web_server.py \
    --num-devices 8 \
    --max-devices 8 \
    --crashed-device-restart \
    --concurrency-limit 8
```

### 3. Run Training

```bash
bash run_hrpo_stepwise.sh
```

Or with custom config:

```bash
export HYDRA_FULL_ERROR=1
python -m hammer_trainer_stepwise.main_ppo \
    --config-path=./scripts \
    --config-name=config_stepwise_32.yaml \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-VL-7B-Instruct" \
    trainer.total_epochs=20 \
    trainer.n_gpus_per_node=8
```

## Project Structure

```
ADMIRE/
├── src/
│   ├── hammer_agent/              # Agent implementation
│   ├── hammer_server/             # Environment server (Gradio)
│   ├── hammer_trainer/            # Base trainer
│   └── hammer_trainer_stepwise/   # Stepwise RL trainer with milestone rewards
├── scripts/                       # Training configs
│   ├── config_stepwise_32.yaml    # Default stepwise config
│   └── config_grpo.yaml           # GRPO config
├── 3rdparty/
│   ├── verl/                      # RL training framework
│   └── android_world/             # Android environment
└── notebooks/
    └── visualize_step.ipynb       # Trajectory visualization
```

## Configuration

Key parameters in `scripts/config_stepwise_32.yaml`:

```yaml
# Environment
env:
  src: [" "]                       
  max_envs: [16]
  max_steps: 30

# Model
actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-VL-7B-Instruct"
  rollout:
    n: 8                           # Number of rollouts per prompt

# Training
trainer:
  total_epochs: 20
  n_gpus_per_node: 8

# Milestone Reward
milestone_reward:
  enable: true
  threshold: 0.75                  # Similarity threshold for matching
  weight: 0.3

process_reward:
  enable: true
  weight: 1.0
  decay_gamma: 0.99
```

## Reward Components

The total reward is computed as:

$$\mathcal{R}_{total} = \mathcal{R}_{outcome} + \eta \cdot \mathcal{R}_{format} + \lambda(t) \cdot \mathcal{R}_{mil}$$

Configure via:

```yaml
milestone_reward:
  enable: true
  weight: 0.3
  strategy: "mix"

process_reward:
  enable: true
  weight: 1.0
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- [verl](https://github.com/volcengine/verl) - ByteDance Seed Team
- [AndroidWorld](https://github.com/google-research/android_world) - Google Research
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Alibaba


