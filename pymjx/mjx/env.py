from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box, Dict
import numpy as np
import mjx._mjx as _mjx


class RLlibMahjongEnv(MultiAgentEnv):
    NUM_ACTION = 181
    NUM_FEATURE = 34 * 4
    ACTION_EMBED_SIZE = 1
    AGENT_OBS_SPACE = Dict({
        "action_mask": Box(0, 1, shape=(NUM_ACTION,)),
        "avail_actions": Box(-10, 10, shape=(NUM_ACTION, ACTION_EMBED_SIZE)),
        "real_obs": Box(0, 1, shape=(NUM_FEATURE,)),
    })
    AGENT_ACTION_SPACE = Discrete(NUM_ACTION)

    def __init__(self, seed=None):
        self.env = _mjx.RLlibMahjongEnv()
        self.action_embeds = [np.random.randn(self.ACTION_EMBED_SIZE) for _ in range(self.NUM_ACTION)]
        self.legal_actions = {}
        if seed is not None:
            self.seed(seed)

    def _make_observation(self, orig_obs_dict):
        obs_dict = {}
        self.legal_actions = {}
        for player_id in orig_obs_dict:
            obs_dict[player_id] = {}
            obs = orig_obs_dict[player_id]
            mask = obs.action_mask()
            obs_dict[player_id]["action_mask"] = np.array(mask)
            obs_dict[player_id]["avail_actions"] = np.array([self.action_embeds[i] if mask[i]
                                                             else np.zeros((self.ACTION_EMBED_SIZE,))
                                                             for i in range(self.NUM_ACTION)])
            obs_dict[player_id]["real_obs"] = np.array(obs.to_feature("small_v0"))
            self.legal_actions[player_id] = obs.legal_actions()
        return obs_dict

    def reset(self):
        orig_obs_dict = self.env.reset()
        return self._make_observation(orig_obs_dict=orig_obs_dict)

    def step(self, orig_act_dict):
        act_dict = {}
        for player_id in orig_act_dict:
            assert player_id in self.legal_actions
            act_dict[player_id] = _mjx.Action(orig_act_dict[player_id], self.legal_actions[player_id])
        orig_obs_dict, orig_rew, orig_done, orig_info = self.env.step(act_dict)
        rew, done, info = {}, {}, {}
        for id in orig_obs_dict.keys():
            rew[id] = orig_rew[id]
            done[id] = orig_done[id]
            info[id] = orig_info[id]
        done["__all__"] = orig_done["__all__"]
        return self._make_observation(orig_obs_dict=orig_obs_dict), rew, done, info

    def seed(self, game_seed):
        self.env.seed(game_seed)


if __name__ == '__main__':
    env = RLlibMahjongEnv()
    env.reset()
    print(env.step({player_id: 1 for player_id, actions in env.legal_actions.items()}))
