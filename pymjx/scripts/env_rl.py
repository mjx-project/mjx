from typing import List

from mjx.env import RLlibMahjongEnv
import mjxproto
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.logger import pretty_print
import gym
import numpy as np


ray.init()

# RLlibMahjongEnv code
def feature_extraction(observation: mjxproto.Observation) -> List[int]:
    return []


def decode_action(action: int) -> mjxproto.Action:
    return mjxproto.Action()


# original multi-agent code
class MyMultiEnv(MultiAgentEnv):
    def __init__(self, config):
        self.player1 = "player1"
        self.player2 = "player2"

    def reset(self):
        return {self.player1: 0, self.player2: 1}

    def step(self, action):
        return {self.player1: 0, self.player2: 1}, {self.player1: 0, self.player2: 1}, {"__all__": False}, {}


config = {
    "env": MyMultiEnv,
    "gamma": 0.9,
    "num_gpus": 0,
    "num_workers": 0,
    "multiagent": {
        "policies": {
            "default_policy": (None, gym.spaces.Discrete(3), gym.spaces.Discrete(3),
                            {}),
        },
        "policy_mapping_fn":
            lambda agent_id: "default_policy"
    },
}
trainer = ppo.PPOTrainer(config=config)

for i in range(1):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 1 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)


# # original gymenv code
# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = gym.spaces.Discrete(3)
#         self.observation_space = gym.spaces.Box(
#             np.array([-1], dtype=np.float32),
#             np.array([1], dtype=np.float32),
#             dtype=np.float32
#         )
#
#     def reset(self):
#         return np.zeros(1)
#
#     def step(self, action):
#         return np.zeros(1), 1, False, {}
#
#
# trainer = ppo.PPOTrainer(env=MyEnv, config={
#     "env_config": {},  # config to pass to env class
# })
#
# for i in range(1):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#
#    if i % 1 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)


# sample code
# config = ppo.DEFAULT_CONFIG.copy()
# config["num_gpus"] = 0
# config["num_workers"] = 1
# trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
#
# for i in range(1):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#
#    if i % 1 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)
#
# env = gym.make("CartPole-v0")
# observation = env.reset()
#
# for i in range(10):
#     print(env.step(trainer.compute_action(observation)))
#     # env.render()
