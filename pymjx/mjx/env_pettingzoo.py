import gym
import numpy as np


class PettingZooMahjongEnv:
    def __init__(self):
        import mjx._mjx as _mjx

        self.env = _mjx.PettingZooMahjongEnv()

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
        mask = obs.action_mask()
        obs_dict = {}
        obs_dict["action_mask"] = np.array(mask, dtype=np.float32)
        obs_dict["real_obs"] = np.array(obs.to_feature("small_v0"), dtype=np.float32)
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