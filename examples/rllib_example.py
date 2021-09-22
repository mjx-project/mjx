from typing import Dict, List

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

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


def train():
    import re

    import numpy as np
    import ray
    import ray.rllib.agents.pg as pg
    from gym.spaces import Box, Dict, Discrete
    from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils.framework import try_import_tf, try_import_torch
    from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
    from ray.tune.logger import pretty_print
    from ray.tune.registry import register_env

    import mjx.env

    tf1, tf, tfv = try_import_tf()
    torch, nn = try_import_torch()

    env = RLlibMahjongEnv()

    class TorchRLlibMahjongEnvModel(DQNTorchModel):
        """PyTorch version of above ParametricActionsModel."""

        def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            true_obs_shape=(env.num_features,),
            action_embed_size=1,
            **kw,
        ):
            DQNTorchModel.__init__(
                self, obs_space, action_space, num_outputs, model_config, name, **kw
            )

            self.action_embed_model = TorchFC(
                Box(-1, 1, shape=true_obs_shape),
                action_space,
                action_embed_size,
                model_config,
                name + "_action_embed",
            )

        def forward(self, input_dict, state, seq_lens):
            # Extract the available actions tensor from the observation.
            action_mask = input_dict["obs"]["action_mask"]

            # Compute the predicted action embedding
            action_embed, _ = self.action_embed_model({"obs": input_dict["obs"]["real_obs"]})

            # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
            # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
            intent_vector = torch.unsqueeze(action_embed, 1)
            action_mask_plus = torch.unsqueeze(action_mask, 2)

            # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
            action_logits = torch.sum(action_mask_plus * intent_vector, dim=2)

            # Mask out invalid actions (use -inf to tag invalid).
            # These are then recognized by the EpsilonGreedy exploration component
            # as invalid actions that are not to be chosen.
            inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

            return action_logits + inf_mask, state

        def value_function(self):
            return self.action_embed_model.value_function()

    class RuleBased(Policy):
        def get_initial_state(self):
            return [np.random.choice(range(env.num_actions))]

        def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs,
        ):
            legal_actions = [idx for idx in range(env.num_actions) if obs_batch[0][idx]]
            if 179 in legal_actions:
                legal_actions.remove(179)
            action = np.random.choice(legal_actions) if legal_actions else 0
            return np.array([action]), state_batches, {}

    def select_policy(agent_id):
        num = re.sub(r"\D", "", agent_id)
        return f"random{num}" if num != "0" else "learned"

    register_env("rllibmahjong", lambda _: RLlibMahjongEnv())
    config = dict(
        {
            "env": "rllibmahjong",
            "model": {
                "custom_model": TorchRLlibMahjongEnvModel,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 0,
            "output": "batch/onpolicy",
            "explore": True,
            # "exploration_config": {
            #     "type": "EpsilonGreedy",
            #     "initial_epsilon": 0.3,
            #     "final_epsilon": 0.01,
            # },
            "exploration_config": {
                "type": "SoftQ",
                "temperature": 1,
            },
            # "gamma": 0.99,
            # "train_batch_size": 2000,
            # "batch_mode": "complete_episodes",
            "multiagent": {
                "policies_to_train": ["learned"],
                "policies": {
                    "learned": (
                        None,
                        env.observation_space,
                        env.action_space,
                        {
                            "framework": "torch",
                        },
                    ),
                    "random1": (RuleBased, env.observation_space, env.action_space, {}),
                    "random2": (RuleBased, env.observation_space, env.action_space, {}),
                    "random3": (RuleBased, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": select_policy,
            },
            "framework": "torch",
        }
    )

    ray.init()
    trainer_obj = pg.PGTrainer(config=config)
    results = []
    for _ in range(10):
        result = trainer_obj.train()
        print(pretty_print(result))


if __name__ == "__main__":
    random_run()
    train()
