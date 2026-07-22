# /// script
# dependencies = [
#   "contrastive-rl-pytorch",
#   "discrete-continuous-embed-readout>=0.2.1",
#   "fire",
#   "gymnasium[box2d]",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.10",
#   "x-mlps-pytorch>=0.3.0",
#   "hl-gauss-pytorch>=0.2.2",
#   "tqdm"
# ]
# ///

from __future__ import annotations

import os
from fire import Fire
from shutil import rmtree
from collections import deque
from functools import partial

import torch
from torch import nn, from_numpy, cat, tensor
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from tqdm import tqdm
import gymnasium as gym
from accelerate import Accelerator

from memmap_replay_buffer import ReplayBuffer

from contrastive_rl_pytorch import (
    ContrastiveRLTrainer,
    ActorTrainer,
    ContrastiveLearning,
    SigmoidContrastiveLearning,
    sample_random_state
)

from einops.layers.torch import Rearrange
from x_mlps_pytorch import ResidualNormedMLP, AttnResidualNormedMLP
from discrete_continuous_embed_readout import Readout

from hl_gauss_pytorch import HLGaussLoss

from dashboard import Dashboard

from train_util import train, exists, default, divisible_by, module_device, CriticWrapper, ConstantGoalWrapper

# main

def main(
    num_episodes = 50_000,
    max_timesteps = 500,
    num_episodes_before_learn = 128,
    buffer_size = 512,
    video_folder = './recordings',
    render_every_eps = None,
    dim_contrastive_embed = 64,
    cl_train_steps = 2_500,
    cl_batch_size = 64,
    actor_batch_size = 128,
    actor_num_train_steps = 1000,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    actor_dim = 64,
    actor_depth = 4,
    critic_dim = 64,
    critic_depth = 8,
    goal_dim = 64,
    goal_depth = 8,
    weight_decay = 1e-4,
    max_grad_norm = 0.5,
    repetition_factor = 2,
    use_sigmoid_contrastive_learning = True,
    sigmoid_bias = -5.,
    cl_l2norm_embed = True,
    exploration_random_goal_prob = 0.025,
    exploration_sample_from_buffer_prob = 0.5,
    reward_part_of_goal = False,
    reward_norm = 100.,
    use_hl_gauss_critic_actions = True,
    hl_gauss_num_bins = 16,
    hl_gauss_sigma = None,
    use_attn_residual_mlp = True,
    use_wandb = False,
    cpu = False
):
    # clear video folder

    rmtree(video_folder, ignore_errors = True)
    os.makedirs(video_folder, exist_ok = True)

    # accelerator

    accelerator = Accelerator(
        log_with = 'wandb' if use_wandb else None,
        cpu = cpu
    )

    if use_wandb:
        accelerator.init_trackers(
            project_name = 'contrastive-rl',
            config = locals()
        )

    # env

    env = gym.make('MountainCarContinuous-v0', render_mode = 'rgb_array')

    # recording

    render_every_eps = default(render_every_eps, num_episodes_before_learn)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'mountaincar',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    dim_state = 2
    dim_goal = 1 + (1 if reward_part_of_goal else 0)
    dim_action = 1

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-mountaincar-continuous',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            state = ('float', dim_state),
            action = ('float', dim_action),
            reward = ('float', 1),
        ),
        circular = True,
        overwrite = True
    )

    # model

    device = accelerator.device

    if use_attn_residual_mlp:
        MLP = AttnResidualNormedMLP
    else:
        MLP = partial(ResidualNormedMLP, residual_every = 4, keel_post_ln = True)

    actor_encoder = nn.Sequential(
        MLP(
            dim_in = dim_state + dim_goal, # state and goal
            dim = actor_dim,
            depth = actor_depth,
            dim_out = dim_action * 2 # for squashed gaussian mu and logvar
        ),
        Rearrange('... (action mu_logvar) -> ... action mu_logvar', mu_logvar = 2)
    ).to(device)

    actor_readout = Readout(
        num_continuous = dim_action,
        continuous_dist_type = 'gaussian',
        continuous_squashed = True,
        dim = 0
    )

    hl_gauss = None
    critic_dim_action = dim_action

    if use_hl_gauss_critic_actions:
        hl_gauss = HLGaussLoss(
            min_value = -1.,
            max_value = 1.,
            num_bins = hl_gauss_num_bins,
            sigma = hl_gauss_sigma,
            clamp_to_range = True
        ).to(device)
        critic_dim_action = dim_action * hl_gauss_num_bins

    critic_encoder = MLP(
        dim_in = dim_state + critic_dim_action,
        dim = critic_dim,
        dim_out = dim_contrastive_embed,
        depth = critic_depth
    ).to(device)

    critic_encoder = CriticWrapper(critic_encoder, hl_gauss, dim_action)

    goal_encoder = MLP(
        dim_in = dim_goal,
        dim = goal_dim,
        dim_out = dim_contrastive_embed,
        depth = goal_depth
    ).to(device)

    # contrastive learning module

    if use_sigmoid_contrastive_learning:
        contrastive_learn = SigmoidContrastiveLearning(bias = sigmoid_bias, l2norm_embed = cl_l2norm_embed)
    else:
        contrastive_learn = ContrastiveLearning(l2norm_embed = True, learned_temp = True)

    state_to_goal_fn = lambda s: s[...,0:1]

    critic_trainer = ContrastiveRLTrainer(
        critic_encoder,
        goal_encoder,
        batch_size = cl_batch_size,
        learning_rate = critic_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        repetition_factor = repetition_factor,
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        state_to_goal_fn = state_to_goal_fn
    )

    # assertions

    assert num_episodes_before_learn > cl_batch_size

    def sample_fn(logits, differentiable = False):
        return actor_readout.sample(logits, differentiable = differentiable, rescale_range = (-1., 1.))

    actor_trainer = ActorTrainer(
        actor_encoder,
        critic_encoder,
        goal_encoder,
        batch_size = actor_batch_size,
        learning_rate = actor_learning_rate,
        weight_decay = weight_decay,
        max_grad_norm = max_grad_norm,
        softmax_actor_output = False,
        reward_part_of_goal = reward_part_of_goal,
        reward_norm = reward_norm,
        cpu = cpu,
        contrastive_learn = contrastive_learn,
        state_to_goal_fn = state_to_goal_fn
    )

    actor_goal = tensor([0.45], device = device)

    if reward_part_of_goal:
        max_reward = tensor([1.], device = device, dtype = torch.float32)
        actor_goal = cat((actor_goal, max_reward), dim = -1)

    # episodes

    dashboard = Dashboard(
        num_episodes,
        title = "Contrastive RL - MountainCar (Continuous)",
        env_name = "MountainCarContinuous - v0",
        hyperparams = dict(
            critic_learning_rate = critic_learning_rate,
            actor_learning_rate = actor_learning_rate,
            cl_batch_size = cl_batch_size,
            actor_batch_size = actor_batch_size,
            buffer_size = f"{buffer_size}",
            max_timesteps = f"{max_timesteps}",
            weight_decay = f"{weight_decay}",
            max_grad_norm = f"{max_grad_norm}",
            repetition_factor = f"{repetition_factor}",
            use_sigmoid_contrastive_learning = use_sigmoid_contrastive_learning,
            exploration_random_goal_prob = exploration_random_goal_prob,
            exploration_sample_from_buffer_prob = exploration_sample_from_buffer_prob,
            use_hl_gauss_critic_actions = use_hl_gauss_critic_actions,
            hl_gauss_num_bins = hl_gauss_num_bins,
            hl_gauss_sigma = f"{hl_gauss_sigma}" if hl_gauss_sigma else "None",
            use_attn_residual_mlp = use_attn_residual_mlp
        )
    )

    train(dashboard,
          accelerator,
          num_episodes,
          ConstantGoalWrapper(env, actor_goal),
          exploration_random_goal_prob,
          replay_buffer,
          exploration_sample_from_buffer_prob,
          device,
          reward_part_of_goal,
          dim_state,
          max_timesteps,
          actor_encoder,
          sample_fn,
          critic_trainer,
          actor_trainer,
          cl_train_steps,
          actor_num_train_steps,
          num_episodes_before_learn,
          use_wandb,
          state_to_goal_fn=state_to_goal_fn,
          log_success=True,
          success_predicate=lambda s: s[0] >= 0.45)

# fire

if __name__ == '__main__':
    Fire(main)
