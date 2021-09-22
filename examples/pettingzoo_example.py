from typing import Dict, List, Optional

import gym
import numpy as np
from pettingzoo import AECEnv

import mjx._mjx as _mjx


class PettingZooMahjongEnv(AECEnv):
    def __init__(self):

        super(PettingZooMahjongEnv, self).__init__()
        self.env = _mjx.PettingZooMahjongEnv()

        self.legal_actions = []

        # consts
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 34 * 10  # TODO: use val from self.env
        self.observation_spaces = {
            i: gym.spaces.Dict(
                {
                    "action_mask": gym.spaces.Box(0, 1, shape=(self.num_actions,)),
                    "real_obs": gym.spaces.Box(0, 1, shape=(self.num_features,)),
                }
            )
            for i in self.possible_agents
        }
        self.action_spaces = {
            i: gym.spaces.Discrete(self.num_actions) for i in self.possible_agents
        }

        # member variables
        self.agents = self.possible_agents
        self.rewards = {i: 0 for i in self.possible_agents}
        self._cumulative_rewards = {i: 0 for i in self.possible_agents}
        self.dones = {i: False for i in self.possible_agents}
        self.infos = {i: {} for i in self.possible_agents}

    @staticmethod
    def _convert_obs(obs):
        mask = obs.action_mask()
        obs_dict = {}
        obs_dict["action_mask"] = np.array(mask, dtype=np.float32)
        obs_dict["real_obs"] = np.array(obs.to_feature("small_v0"), dtype=np.float32)
        return obs_dict

    def _update_legal_actions(self, obs):
        self.legal_actions = obs.legal_actions()

    def reset(self):
        self.env.reset()

        # reset member varialbes
        self.agents = self.possible_agents
        self.rewards = {i: 0 for i in self.possible_agents}
        self._cumulative_rewards = {i: 0 for i in self.possible_agents}
        self.dones = {i: False for i in self.possible_agents}
        self.infos = {i: {} for i in self.possible_agents}

        obs = self.observe(self.agent_selection)
        self._update_legal_actions(obs)

    def last(self, observe=True):
        obs, cumulative_reward, done, info = super().last(observe)
        if observe:
            obs = self._convert_obs(obs)
        return obs, cumulative_reward, done, info

    def step(self, action: Optional[int]):
        if self.dones[self.agent_selection]:
            self._was_done_step(action)
        if action is None:
            # set dummy action
            action = 180
        self.env.step(_mjx.Action(action, self.legal_actions))
        if self.agent_selection is not None:
            obs, _, done, _ = self.env.last(True)
            if done and not self.dones[self.agent_selection]:
                rewards = self.env.rewards()
                self.rewards = {i: rewards[i] for i in self.possible_agents}
                self.dones = {i: True for i in self.possible_agents}
                self.infos = {i: {} for i in self.possible_agents}
            self._update_legal_actions(obs)
        self._accumulate_rewards()

    def seed(self, seed):
        self.env.seed(seed)

    def observe(self, agent):
        return self.env.observe(agent)

    def agents(self):
        return self.env.agents()

    @property
    def possible_agents(self):
        return self.env.possible_agents()

    @property
    def agent_selection(self):
        return self.env.agent_selection()

    @agent_selection.setter
    def agent_selection(self, val):
        pass


def random_run():
    import random

    random.seed(1234)
    env = PettingZooMahjongEnv()
    env.seed(1234)
    env.reset()
    results = []
    for agent in env.agent_iter():
        observation, reward, done, info = env.last(True)
        results.append((agent, observation, reward, done, info))
        legal_actions = [i for i, b in enumerate(observation["action_mask"]) if b]
        action = random.choice(legal_actions)
        env.step(action if not done else None)

    # PettingZoo Test
    # https://github.com/mjx-project/mjx/pull/887
    # Last iteration before done
    agent, observation, reward, done, info = results[-5]
    assert observation["action_mask"][180] == 0
    assert reward == 0
    assert not done
    assert info == {}

    # After done
    final_rewards = set([])

    agent, observation, reward, done, info = results[-4]
    assert observation["action_mask"][180] == 1
    final_rewards.add(reward)
    assert done
    assert info == {}

    agent, observation, reward, done, info = results[-3]
    assert observation["action_mask"][180] == 1
    final_rewards.add(reward)
    assert done
    assert info == {}

    agent, observation, reward, done, info = results[-2]
    assert observation["action_mask"][180] == 1
    final_rewards.add(reward)
    assert done
    assert info == {}

    agent, observation, reward, done, info = results[-1]
    assert observation["action_mask"][180] == 1
    final_rewards.add(reward)
    assert done
    assert info == {}

    assert final_rewards == {-135, 0, 45, 90}


def pettingzoo_api_test():
    from pettingzoo.test import api_test

    env = PettingZooMahjongEnv()
    api_test(env, num_cycles=100000, verbose_progress=False)


if __name__ == "__main__":
    random_run()
    pettingzoo_api_test()
