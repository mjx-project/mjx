from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gym
from typing import Dict, List

import mjx._mjx as _mjx


class RLlibMahjongEnv(MultiAgentEnv):
    def __init__(self):

        self.env = _mjx.RLlibMahjongEnv()

        self.legal_actions = {}

        # consts
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 34 * 10  # TODO: use val from self.env
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.num_actions,)),
                "real_obs": gym.spaces.Box(0, 1, shape=(self.num_features,)),
            }
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

    @staticmethod
    def _convert_obs(obs):
        obs_dict = {}
        for player_id, obs in obs.items():
            mask = obs.action_mask()
            obs_dict[player_id] = {}
            obs_dict[player_id]["action_mask"] = np.array(mask, dtype=np.float32)
            obs_dict[player_id]["real_obs"] = np.array(
                obs.to_feature("small_v0"), dtype=np.float32
            )
        return obs_dict

    def _update_legal_actions(self, obs):
        self.legal_actions = {}
        for player_id, obs in obs.items():
            self.legal_actions[player_id] = obs.legal_actions()

    def reset(self):
        obs = self.env.reset()
        self._update_legal_actions(obs)
        return RLlibMahjongEnv._convert_obs(obs)

    def step(self, action_dict):

        act_dict = {}
        for player_id, action in action_dict.items():
            assert player_id in self.legal_actions
            act_dict[player_id] = _mjx.Action(action, self.legal_actions[player_id])
        obs, rewards, dones, infos = self.env.step(act_dict)
        self._update_legal_actions(obs)
        return RLlibMahjongEnv._convert_obs(obs), rewards, dones, infos

    def seed(self, seed):
        self.env.seed(seed)


def random_run():
    import random
    import mjx

    random.seed(1234)
    env = RLlibMahjongEnv()
    env.seed(1234)
    obs_dict = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        action_dict = {}
        for agent, obs in obs_dict.items():
            legal_actions = [i for i, b in enumerate(obs["action_mask"]) if b]
            action_dict[agent] = random.choice(legal_actions)
        obs_dict, rewards, dones, info = env.step(action_dict)
    assert len(rewards) == 4
    assert list(sorted(rewards.values())) == [-135, 0, 45, 90]


if __name__ == '__main__':
	random_run()