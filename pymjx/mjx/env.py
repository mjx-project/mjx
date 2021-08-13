import gym
import numpy as np


class RLlibMahjongEnv:
    def __init__(self):
        import mjx._mjx as _mjx

        self.env = _mjx.RLlibMahjongEnv()

        self.legal_actions = {}

        # consts
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 34 * 4  # TODO: use val from self.env
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


class PettingZooMahjongEnv:
    def __init__(self):
        import mjx._mjx as _mjx

        self.env = _mjx.PettingZooMahjongEnv()

        self.legal_actions = []

        # consts
        self.num_actions = 181  # TODO: use val from self.env
        self.num_features = 34 * 4  # TODO: use val from self.env
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.num_actions,)),
                "real_obs": gym.spaces.Box(0, 1, shape=(self.num_features,)),
            }
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

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
        obs = self.env.last(True)[0]
        self._update_legal_actions(obs)

    def last(self, observe):
        obs, reward, done, info = self.env.last(True)
        obs = self._convert_obs(obs)
        return obs, reward, done, info

    def step(self, action: int):
        import mjx._mjx as _mjx

        self.env.step(_mjx.Action(action, self.legal_actions))
        obs = self.env.last(True)[0]
        self._update_legal_actions(obs)

    def seed(self, seed):
        self.env.seed(seed)

    def observe(self, agent):
        return self.env.observe(agent)

    def agents(self):
        return self.env.agents()

    def possible_agents(self):
        return self.env.possible_agents()

    def agent_selection(self):
        return self.env.agent_selection()

    def agent_iter(self, max_iter=2**63):
        count = 0
        done_count = 0
        while done_count < 5 and count < max_iter:
            yield self.agent_selection()
            count += 1
            done_count += self.last(False)[2]