import random
from typing import Dict, List, Optional

import gym
import numpy as np
from pettingzoo import AECEnv

import mjx


class MjxEnv:
    def __init__(
        self,
        player_ids: List[str] = ["player_0", "player_1", "player_2", "player_3"],
        observe_all: bool = False,
    ):
        import mjx._mjx as _mjx

        self._env = _mjx.MjxEnv(player_ids, observe_all)

    def seed(self, seed) -> None:
        self._env.seed(seed)

    def reset(self) -> Dict[str, mjx.Observation]:
        cpp_obs_dict = self._env.reset()
        return {k: mjx.Observation(v) for k, v in cpp_obs_dict.items()}

    def step(self, aciton_dict: Dict[str, mjx.Action]) -> Dict[str, mjx.Observation]:
        cpp_action_dict = {k: v._cpp_obj for k, v in aciton_dict.items()}
        cpp_obs_dict = self._env.step(cpp_action_dict)
        return {k: mjx.Observation(v) for k, v in cpp_obs_dict.items()}

    def done(self) -> bool:
        return self._env.done()

    def rewards(self) -> Dict[str, int]:
        return self._env.rewards()

    @property
    def state(self) -> mjx.State:
        return mjx.State(self._env.state())


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


class PettingZooMahjongEnv(AECEnv):
    def __init__(self):
        import mjx._mjx as _mjx

        super(PettingZooMahjongEnv, self).__init__()
        self.env = _mjx.PettingZooMahjongEnv()

        self.legal_actions = []

        # consts
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 34 * 4  # TODO: use val from self.env
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
        import mjx._mjx as _mjx

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
