import random

import gym
import numpy as np


class SingleAgentEnv(gym.Env):
    """player_0 is controllable. Other player is random agents."""

    def __init__(self):
        super().__init__()
        self.env = RLlibMahjongEnv()
        self.action_dict = {}

        # consts
        self.agent_id = "player_0"
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 340  # TODO: use val from self.env
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.num_actions,)),
                "real_obs": gym.spaces.Box(0, 1, shape=(self.num_features,)),
            }
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

    def reset(self):
        obs_dict = self.env.reset()
        while True:
            self.action_dict = {}
            for agent, obs in obs_dict.items():
                if agent == self.agent_id:
                    continue
                self.action_dict[agent] = self.random_action(obs)

            if self.agent_id in obs_dict:
                return obs
            else:
                obs_dict, rewards, dones, infos = self.env.step(self.action_dict)

    def step(self, action):
        self.action_dict[self.agent_id] = action
        obs_dict, rewards, dones, infos = self.env.step(self.action_dict)
        while True:
            self.action_dict = {}
            for agent, obs in obs_dict.items():
                if agent == self.agent_id:
                    continue
                self.action_dict[agent] = self.random_action(obs)

            if self.agent_id in obs_dict:
                return (
                    obs_dict[self.agent_id],
                    rewards[self.agent_id],
                    dones[self.agent_id],
                    infos[self.agent_id],
                )
            else:
                obs_dict, rewards, dones, infos = self.env.step(self.action_dict)

    def random_action(self, obs):
        legal_actions = [i for i, b in enumerate(obs["action_mask"]) if b]
        return random.choice(legal_actions)


class RLlibMahjongEnv:
    def __init__(self):
        import mjx._mjx as _mjx

        self.env = _mjx.RLlibMahjongEnv()

        self.legal_actions = {}

        # consts
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 340  # TODO: use val from self.env
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
        import mjx._mjx as _mjx

        act_dict = {}
        for player_id, action in action_dict.items():
            assert player_id in self.legal_actions
            act_dict[player_id] = _mjx.Action(action, self.legal_actions[player_id])
        obs, rewards, dones, infos = self.env.step(act_dict)
        self._update_legal_actions(obs)
        return RLlibMahjongEnv._convert_obs(obs), rewards, dones, infos

    def seed(self, seed):
        self.env.seed(seed)
