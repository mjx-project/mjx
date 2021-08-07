from gym.spaces import Discrete, Box, Dict
import numpy as np


class RLlibMahjongEnv:
    NUM_ACTION = 181
    NUM_FEATURE = 34 * 4
    ACTION_EMBED_SIZE = 1
    AGENT_OBS_SPACE = Dict({
        "action_mask": Box(0, 1, shape=(NUM_ACTION,)),
        "real_obs": Box(0, 1, shape=(NUM_FEATURE,)),
    })
    AGENT_ACTION_SPACE = Discrete(NUM_ACTION)

    def __init__(self):
        import mjx._mjx as _mjx
        self.env = _mjx.RLlibMahjongEnv()
        self.legal_actions = {}

    @staticmethod
    def _make_observation(orig_obs_dict):
        obs_dict = {}
        for player_id, obs in orig_obs_dict.items():
            obs_dict[player_id] = {}
            mask = obs.action_mask()
            obs_dict[player_id]["action_mask"] = np.array(mask)
            obs_dict[player_id]["real_obs"] = np.array(
                obs.to_feature("small_v0"))
        return obs_dict

    def _update_legal_actions(self, orig_obs_dict):
        self.legal_actions = {}
        for player_id, obs in orig_obs_dict.items():
            self.legal_actions[player_id] = obs.legal_actions()

    def reset(self):
        orig_obs_dict = self.env.reset()
        self._update_legal_actions(orig_obs_dict)
        return RLlibMahjongEnv._make_observation(orig_obs_dict=orig_obs_dict)

    def step(self, orig_act_dict):
        import mjx._mjx as _mjx
        act_dict = {}
        for player_id, action in orig_act_dict.items():
            assert player_id in self.legal_actions
            act_dict[player_id] = _mjx.Action(
                action, self.legal_actions[player_id])
        orig_obs_dict, orig_rew, orig_done, orig_info = self.env.step(act_dict)
        self._update_legal_actions(orig_obs_dict)
        return RLlibMahjongEnv._make_observation(orig_obs_dict=orig_obs_dict), orig_rew, orig_done, orig_info

    def seed(self, seed):
        self.env.seed(seed)
