import re
import _mjx
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.impala as impla
import ray.rllib.agents.marwil as marwil
import ray.rllib.agents.pg as pg
import ray.rllib.agents.a3c as a3c
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gym.spaces import Discrete, Box, Dict
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy
from ray import tune
import matplotlib.pylab as plt
import time

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class WrappedMahjongEnv(MultiAgentEnv):
    PLAYER_IDS = ["player_0", "player_1", "player_2", "player_3"]
    NUM_ACTION = 181
    NUM_FEATURE = 34 * 4
    ACTION_EMBED_SIZE = 1
    AGENT_OBS_SPACE = Dict({
        "action_mask": Box(0, 1, shape=(NUM_ACTION,)),
        "avail_actions": Box(-10, 10, shape=(NUM_ACTION, ACTION_EMBED_SIZE)),
        "real_obs": Box(0, 1, shape=(NUM_FEATURE,)),
    })
    AGENT_ACTION_SPACE = Discrete(NUM_ACTION)

    def __init__(self, env, seed=None):
        self.env = env
        self.action_embeds = [np.random.randn(self.ACTION_EMBED_SIZE) for _ in range(self.NUM_ACTION)]
        self.legal_actions = {}
        if seed is not None:
            self.seed(seed)
        self.learn_dict = {}

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
            obs_dict[player_id]["real_obs"] = np.array(obs.feature("small_v0"))
            self.legal_actions[player_id] = obs.legal_actions()
        return obs_dict

    def reset(self):
        orig_obs_dict = self.env.reset()
        for player_id in self.PLAYER_IDS:
            self.learn_dict[player_id] = {}
            for action in range(self.NUM_ACTION):
                self.learn_dict[player_id][action] = 0
        return self._make_observation(orig_obs_dict=orig_obs_dict)

    def step(self, orig_act_dict):
        act_dict = {}
        # print(orig_act_dict)
        for player_id in self.PLAYER_IDS:
            if player_id in self.legal_actions:
                act_dict[player_id] = _mjx.Action(orig_act_dict[player_id], self.legal_actions[player_id])
                self.learn_dict[player_id][orig_act_dict[player_id]] += 1
        # print(act_dict)
        orig_obs_dict, orig_rew, orig_done, orig_info = self.env.step(act_dict)
        rew, done, info = {}, {}, {}
        for id in orig_obs_dict.keys():
            rew[id] = orig_rew[id]
            done[id] = orig_done[id]
            info[id] = orig_info[id]
        done["__all__"] = orig_done["__all__"]
        if done["__all__"]:
            for player_id in self.PLAYER_IDS:
                dict = self.learn_dict[player_id].copy()
                dict = sorted(dict.items())
                x, y = zip(*dict)
                plt.plot(x, y)
            fig.savefig("plot.png")
            plt.clf()
            for player_id in self.PLAYER_IDS:
                dict = self.learn_dict[player_id].copy()
                dict = sorted(dict.items())
                x, y = zip(*dict)
                plt.plot(x[74:], y[74:])
            fig.savefig("plot2.png")
        return self._make_observation(orig_obs_dict=orig_obs_dict), rew, done, info

    def seed(self, game_seed):
        self.env.seed(game_seed)


class TorchRLlibMahjongEnvModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(WrappedMahjongEnv.NUM_FEATURE,),
                 action_embed_size=WrappedMahjongEnv.ACTION_EMBED_SIZE,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_embed_model = TorchFC(
            Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embed")
        self.timestep = 1

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["real_obs"]
        })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = torch.unsqueeze(action_embed, 1)
        action_mask_plus = torch.unsqueeze(action_mask, 2)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = torch.sum(action_mask_plus * intent_vector, dim=2)
        # action_logits = torch.sum(avail_actions * intent_vector, dim=2)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        # action_logits[:, 179] = FLOAT_MIN
        self.timestep += 1

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


class RandomSelect(Policy):
    def get_initial_state(self):
        return [
            np.random.choice(
                range(WrappedMahjongEnv.NUM_ACTION)
            )
        ]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        legal_actions = [idx for idx in range(WrappedMahjongEnv.NUM_ACTION) if obs_batch[0][idx]]
        action = np.random.choice(legal_actions) if legal_actions else 0
        # print(action)
        return np.array([action]), state_batches, {}


class RuleBased(Policy):
    def get_initial_state(self):
        return [
            np.random.choice(
                range(WrappedMahjongEnv.NUM_ACTION)
            )
        ]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        legal_actions = [idx for idx in range(WrappedMahjongEnv.NUM_ACTION) if obs_batch[0][idx]]
        if 179 in legal_actions:
            legal_actions.remove(179)
        action = np.random.choice(legal_actions) if legal_actions else 0
        # print(action)
        return np.array([action]), state_batches, {}


# 直接環境をstepさせ、ランダムポリシーでゲーム進行する
def random_policy():
    env = WrappedMahjongEnv(env=_mjx.RLlibMahjongEnv(), seed=2)
    obs_dict = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        act_dict = {}
        for id, obs in obs_dict.items():
            action_mask = obs_dict[id]["action_mask"]
            legal_actions = [idx for idx in range(len(action_mask)) if action_mask[idx]]
            act_dict[id] = np.random.choice(legal_actions) if legal_actions else 0
        obs_dict, rewards, dones, info = env.step(act_dict)


# RLlibの形式で環境を動かし、ランダムポリシーでゲーム進行する
def rllib_random_policy():

    def select_policy(agent_id):
        num = re.sub(r"\D", "", agent_id)
        return f"random{num}"

    register_env("rllibmahjong", lambda _: WrappedMahjongEnv(env=_mjx.RLlibMahjongEnv(), seed=3))
    config = dict(
        {
            "env": "rllibmahjong",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 0,
            "multiagent": {
                "policies_to_train": ["learned"],
                "policies": {
                    "random0": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random1": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random2": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random3": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {})
                },
                "policy_mapping_fn": select_policy
            },
            "framework": "torch"
        })
    stop = {
        "training_iteration": 10000,
        "timesteps_total": 100000,
    }
    trainer_obj = ppo.PPOTrainer(config=config)
    for _ in range(100):
        results = trainer_obj.train()
        print(pretty_print(results))


# RLlibの形式で環境を動かし、ルールベースドなポリシーでゲーム進行する
def rllib_rulebased():

    def select_policy(agent_id):
        num = int(re.sub(r"\D", "", agent_id))
        if num == 0:
            return "learning"
        elif num == 1:
            return "random1"
        elif num == 2:
            return "random2"
        else:
            return "random3"

    register_env("rllibmahjong", lambda _: WrappedMahjongEnv(env=_mjx.RLlibMahjongEnv(), seed=3))
    config = dict(
        {
            "env": "rllibmahjong",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 0,
            "output": "batch/rulebased",
            "multiagent": {
                "policies_to_train": ["train"],
                "policies": {
                    "learned": (RuleBased,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random1": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random2": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random3": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {})
                },
                "policy_mapping_fn": select_policy
            },
            "framework": "torch"
        })
    trainer_obj = ppo.PPOTrainer(config=config)
    for _ in range(10):
        results = trainer_obj.train()
        print(pretty_print(results))


# 実際に環境と相互作用しながらモデル学習を行う
def online_model_policy():

    def select_policy(agent_id):
        num = re.sub(r"\D", "", agent_id)
        return f"random{num}" if num != "0" else "learned"

    register_env("rllibmahjong", lambda _: WrappedMahjongEnv(env=_mjx.RLlibMahjongEnv(), seed=3))
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
                    "learned": (None, WrappedMahjongEnv.AGENT_OBS_SPACE, WrappedMahjongEnv.AGENT_ACTION_SPACE, {
                        "framework": "torch",
                    }),
                    "random1": (RuleBased,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random2": (RuleBased,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random3": (RuleBased,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {})
                },
                "policy_mapping_fn": select_policy
            },
            "framework": "torch"
        })
    trainer_obj = impla.ImpalaTrainer(config=config)
    results = []
    for _ in range(1000):
        print(_)
        result = trainer_obj.train()
        if "learned" in result["policy_reward_mean"]:
            print(result["policy_reward_mean"]["learned"])
            results.append(result["policy_reward_mean"]["learned"])
        # print(pretty_print(results))
    fig = plt.figure()
    plt.clf()
    plt.plot(list(range(len(results))), results)
    fig.savefig(f"perf_rulebased_{time.time()}.png")


# online_model_policyで集めた経験を使って学習を行う
def offline_model_policy():

    def select_policy(agent_id):
        num = re.sub(r"\D", "", agent_id)
        return f"random{num}" if num != "0" else "learned"

    register_env("rllibmahjong", lambda _: WrappedMahjongEnv(env=_mjx.RLlibMahjongEnv(), seed=3))
    config = dict(
        {
            "env": "rllibmahjong",
            "model": {
                "custom_model": TorchRLlibMahjongEnvModel,
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": 0,
            "num_workers": 0,
            # "input":{
            #     "batch/onpolicy": 0.8,
            #     "sampler": 0.2,
            # },
            # "input_evaluation": [],
            "explore": False,
            "multiagent": {
                "policies_to_train": ["learned"],
                "policies": {
                    "learned": (None, WrappedMahjongEnv.AGENT_OBS_SPACE, WrappedMahjongEnv.AGENT_ACTION_SPACE, {
                        "framework": "torch",
                    }),
                    "random1": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random2": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {}),
                    "random3": (RandomSelect,
                                WrappedMahjongEnv.AGENT_OBS_SPACE,
                                WrappedMahjongEnv.AGENT_ACTION_SPACE,
                                {})
                },
                "policy_mapping_fn": select_policy
            },
            "framework": "torch"
        })
    trainer_obj = ppo.PPOTrainer(config=config)
    for _ in range(50):
        results = trainer_obj.train()
        print(pretty_print(results))


if __name__ == '__main__':
    fig = plt.figure()
    ray.init()
    # random_policy()
    # rllib_random_policy()
    # rllib_rulebased()
    online_model_policy()
    # offline_model_policy()
    ray.shutdown()
