import pfrl
import torch
import torch.nn
import gym
import numpy

#env = gym.make('CartPole-v0')
import mjx.env
env = mjx.env.SingleAgentEnv()

obs_size = env.observation_space["real_obs"].low.size
n_actions = env.action_space.n
print('obs_size:', obs_size)
print('n_actions:', n_actions)

q_func = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)


# Set the discount factor that discounts future rewards.
gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x["real_obs"].astype(numpy.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = -1

from pfrl.agents import dqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward

import collections
import copy
import ctypes
import multiprocessing as mp
import multiprocessing.synchronize
import time
from logging import Logger, getLogger
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import pfrl
from pfrl import agent
from pfrl.action_value import ActionValue
from pfrl.explorer import Explorer
from pfrl.replay_buffer import (
    AbstractEpisodicReplayBuffer,
    ReplayUpdater,
    batch_experiences,
    batch_recurrent_experiences,
)
from pfrl.replay_buffers import PrioritizedReplayBuffer
from pfrl.utils.batch_states import batch_states
from pfrl.utils.contexts import evaluating
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.utils.recurrent import (
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
    recurrent_state_as_numpy,
)

class DoubleDQN(dqn.DQN):
    """Double DQN.
    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch["next_state"]

        with evaluating(self.model):
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = self.model(batch_next_state)

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(next_qout.greedy_actions)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            #batch_argmax = batch_av.greedy_actions.detach().cpu().numpy()

            batch_argmax = []
            for i in range(len(batch_obs)):
                mask = [m == 0 for m in batch_obs[i]["action_mask"]]
                batch_argmax.append(
                        numpy.ma.masked_array(batch_av.q_values[i].detach().cpu().numpy(), mask).argmax()
                        )

            for i in range(len(batch_obs)):
                assert(batch_obs[i]["action_mask"][batch_argmax[i]] == 1)

        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t,
                    lambda: batch_argmax[i],
                    action_value=batch_av[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax
        return batch_action


# Now create an agent that will interact with the environment.
#agent = pfrl.agents.DoubleDQN(
agent = DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)


n_episodes = 300
max_episode_len = 200
for i in range(1, n_episodes + 1):
    print('i:', i)
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)

        assert(obs["action_mask"][action] == 1)

        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')


with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)

# Save an agent to the 'agent' directory
agent.save('agent')

