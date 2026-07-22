from __future__ import annotations
from collections import deque

import torch
from torch import from_numpy, cat
import torch.nn as nn
import gymnasium as gym

from contrastive_rl_pytorch import (
    sample_random_state
)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def module_device(m):
    return next(m.parameters()).device

def identity(t):
    return t

class CriticWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hl_gauss: nn.Module | None,
        dim_action: int
    ):
        super().__init__()
        self.encoder = encoder
        self.hl_gauss = hl_gauss
        self.dim_action = dim_action

    def forward(self, state_and_action):
        if not exists(self.hl_gauss):
            return self.encoder(state_and_action)

        dim_action = self.dim_action

        state, action = state_and_action[..., :-dim_action], state_and_action[..., -dim_action:]

        action_probs = self.hl_gauss.transform_to_probs(action)
        action_probs = rearrange(action_probs, '... a bins -> ... (a bins)')

        state_and_action = cat((state, action_probs), dim = -1)

        return self.encoder(state_and_action)

class ConstantGoalWrapper(gym.Wrapper):
    """ Class that provides function 'sample_goal()' returning a constant goal.
    Use this class in 'train(..)' function if you have a constant goal and your
    environment does not already have a 'sample_goal()' function. """
    def __init__(self, env, goal):
        super().__init__(self, env)
        self.goal = goal

    def sample_goal(self):
        return self.goal

# training routine. 'env' must be gymnasium environment (wrapper) subclass that provides
# a 'sample_goal()' function.
def train(dashboard,
          accelerator,
          num_episodes,
          env,
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
          state_to_goal_fn=identity,
          log_success=False,
          success_predicate=lambda _: False):

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)
    rolling_success = deque(maxlen = 100)

    with dashboard.create_renderable() as live:
        for eps in range(num_episodes):

            state, *_ = env.reset()

            cum_reward = 0.
            eps_steps = 0
            cl_loss = 0.
            actor_loss = 0.

            # decide on goal for the episode

            is_exploring = torch.rand(()) < exploration_random_goal_prob

            eps_goal = env.sample_goal()

            if is_exploring:
                eps_goal = sample_random_state(
                    replay_buffer,
                    env,
                    exploration_sample_from_buffer_prob
                ).to(device)

                eps_goal = state_to_goal_fn(eps_goal)

                if reward_part_of_goal and eps_goal.shape[-1] == dim_state:
                    rand_reward = torch.rand((1,), device = device, dtype = torch.float32)
                    eps_goal = cat((eps_goal, rand_reward), dim = -1)

            states = []
            actions = []
            rewards = []
            success = False

            for _ in range(max_timesteps):

                actor_encoder.eval()

                curr_state = from_numpy(state).to(device)

                action_logits = actor_encoder(cat((curr_state, eps_goal), dim = -1))

                action = sample_fn(action_logits)

                next_state, reward, terminated, truncated, *_ = env.step(action.detach().cpu().numpy())

                # store transition data

                states.append(state)
                actions.append(action.detach().cpu())
                rewards.append(reward)

                cum_reward += reward
                eps_steps += 1

                done = truncated or terminated

                if done:
                    success = success_predicate(next_state)
                    break

                state = next_state

            # store episode if length >= 2

            if len(states) >= 2:
                replay_buffer.store_episode(
                    state = states,
                    action = actions,
                    reward = rewards
                )

            if not is_exploring:
                rolling_reward.append(cum_reward)
                rolling_steps.append(eps_steps)
                rolling_success.append(success)

                dashboard.update_metrics(
                    last_eps_reward = f"{cum_reward:.2f}",
                    last_eps_steps = eps_steps
                )

            live.update(dashboard.render())

            # train the critic and actor

            if (eps + 1) >= num_episodes_before_learn and divisible_by(eps + 1, num_episodes_before_learn):

                data = replay_buffer.get_all_data(
                    fields = ['state', 'action', 'reward'],
                    meta_fields = ['episode_lens']
                )

                trajectories = data['state']
                episode_lens = data['episode_lens']
                actions_for_critic = data['action']
                rewards_for_critic = data['reward']

                # cl_loss[0] is the loss, cl_loss[1] is the sigreg loss, whatever that
                # is meant to be.
                cl_loss = critic_trainer(
                    trajectories,
                    cl_train_steps,
                    lens = episode_lens,
                    actions = actions_for_critic,
                    rewards = rewards_for_critic,
                    pbar = dashboard.critic_pbar
                )[0]

                actor_loss = actor_trainer(
                    trajectories,
                    actor_num_train_steps,
                    lens = episode_lens,
                    rewards = rewards_for_critic,
                    pbar = dashboard.actor_pbar,
                    sample_fn = lambda logits: sample_fn(logits, differentiable = True)
                )

                dashboard.update_metrics(
                    critic_loss = f"{cl_loss:.4f}",
                    actor_loss = f"{actor_loss:.4f}"
                )

            dashboard.advance_progress()

            if not is_exploring:
                avg_reward = sum(rolling_reward) / len(rolling_reward) if len(rolling_reward) > 0 else 0.
                avg_steps = sum(rolling_steps) / len(rolling_steps) if len(rolling_steps) > 0 else 0.
                avg_success = float(sum(rolling_success)) / len(rolling_success) if len(rolling_success) > 0 else 0.

                dashboard.update_metrics(
                    avg_cum_reward_100 = f"{avg_reward:.2f}",
                    avg_steps_100 = f"{avg_steps:.1f}",
                    avg_cum_success_100 = f"{100.0 * avg_success:.1f}%"
                )

                if use_wandb:
                    log_txt = {
                        "avg_cum_reward_100": avg_reward,
                        "avg_steps_100": avg_steps,
                        "last_eps_reward": cum_reward,
                        "critic_loss": cl_loss,
                        "actor_loss": actor_loss
                    }

                    if log_success:
                        log_txt["avg_cum_success_100"] = avg_success

                    accelerator.log(log_txt)

            live.update(dashboard.render())

        if use_wandb:
            accelerator.end_training()