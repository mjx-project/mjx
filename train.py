import pfrl
import torch
import torch.nn
import gym
import numpy


####################
# patch: action mask
####################

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

    def act(self, obs: Any) -> Any:
        random_action_func.set_mask(obs["action_mask"])
        return self.batch_act([obs])[0]

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



from logging import getLogger

import numpy as np

from pfrl import explorer


def select_action_epsilon_greedily(epsilon, random_action_func, greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class ConstantEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with constant epsilon.
    Args:
      epsilon: epsilon used
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, epsilon, random_action_func, logger=getLogger(__name__)):
        assert epsilon >= 0 and epsilon <= 1
        self.epsilon = epsilon
        self.random_action_func = random_action_func
        self.logger = logger

    def select_action(self, t, greedy_action_func, action_value=None):
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        return a

    def __repr__(self):
        return "ConstantEpsilonGreedy(epsilon={})".format(self.epsilon)



###############


#env = gym.make('CartPole-v0')
import mjx.env
env = mjx.env.SingleAgentEnv()


class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions, num_layers, num_units):
        super().__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(
            obs_size, num_units)])
        self.linears.extend([torch.nn.Linear(num_units, num_units) for _ in range(num_layers)])
        self.fc = torch.nn.Linear(num_units, n_actions)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = F.relu(x)
        x = self.fc(x)
        return pfrl.action_value.DiscreteActionValue(x)

obs_size = env.observation_space["real_obs"].low.size
n_actions = env.action_space.n

num_layers = 2
num_units = 2048

#q_func = QFunction(obs_size, n_actions, num_layers, num_units)

# use pretrained model
lr = 1e-3
name = f"small_v0_not_jit-lr={lr}-num_layers={num_layers}-num_units={num_units}"
MODEL_DIR = "/Users/habarakeigo/ghq/github.com/mjx-project/mjx/pymjx/mjx/resources" # FIXME
q_func = QFunction(obs_size, n_actions, num_layers, num_units)
q_func.load_state_dict(torch.load(MODEL_DIR + '/' + name + '.pt'))


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)


# Set the discount factor that discounts future rewards.
gamma = 0.9


class RandomActionFunction:
    def __init__(self, n_actions: int):
        self.actions = list(range(n_actions))
        self.prob = [1.0 / n_actions] * n_actions

    def set_mask(self, mask):
        assert(n_actions == len(mask))
        self.prob = mask / sum(mask)

    def __call__(self):
        return numpy.random.choice(self.actions, p=self.prob)

random_action_func = RandomActionFunction(n_actions)

# Use epsilon-greedy for exploration
#explorer = pfrl.explorers.ConstantEpsilonGreedy(
#    epsilon=0.3, random_action_func=env.action_space.sample)
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=random_action_func)


# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x["real_obs"].astype(numpy.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = -1


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


#n_episodes = 300
n_episodes = 10
#max_episode_len = 200
max_episode_len = 400
avg_results = []
sum_reward = 0
for i in range(1, n_episodes + 1):
    print('i:', i)
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)

        # Assertion: valid action
        assert(obs["action_mask"][action] == 1)

        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    sum_reward += R
    avg_results.append(sum_reward / i)
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')


import matplotlib.pyplot as plt
print('avg_results:', avg_results)
plt.plot(avg_results)
plt.show()


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
            reset = t == max_episode_len
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)

# Save an agent to the 'agent' directory
agent.save('agent')

