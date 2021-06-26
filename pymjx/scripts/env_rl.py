from typing import List, Dict, Tuple
from mjx.env import RLlibMahjongEnv
import mjxproto
from ray.tune.logger import pretty_print
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from ray.rllib.policy.policy import Policy
from gym.spaces import Discrete, Box
import random

from ray.rllib.examples.env.rock_paper_scissors import RockPaperScissors
from ray.rllib.examples.policy.rock_paper_scissors_dummies import AlwaysSameHeuristic
from ray.rllib.policy.view_requirement import ViewRequirement

ray.init()


# original MultiAgentEnv code
class JankenEnv(MultiAgentEnv):
    Guu = 0
    Choki = 1
    Paa = 2

    def __init__(self, config):
        self.player1 = "player1"
        self.player2 = "player2"
        self.observation_space = Discrete(3)
        self.action_space = Discrete(3)
        self.last_moves = None
        self.game_count = None

    def reset(self):
        self.game_count = 0
        self.last_moves = {self.player1: 0, self.player2: 0}
        return self.last_moves

    def _judge_winner(self, action_dict: Dict[str, int]) -> Dict[str, int]:
        if (action_dict[self.player1] - action_dict[self.player2] + 1) % 3 == 0:
            return {self.player1: 1, self.player2: -1}
        elif action_dict[self.player1] == action_dict[self.player2]:
            return {self.player1: 0, self.player2: 0}
        else:
            return {self.player1: -1, self.player2: 1}

    def step(self, action_dict: Dict[str, int]) -> \
            Tuple[Dict[str, int], Dict[str, int], Dict[str, bool], Dict[str, str]]:
        self.last_moves = action_dict.copy()
        observations = self.last_moves.copy()
        rewards = self._judge_winner(action_dict.copy())
        dones = {
            "__all__": self.game_count >= 10,
        }
        self.game_count += 1
        return observations, rewards, dones, {}


class ChangeInOrder(Policy):
    def get_initial_state(self):
        return [
            random.choice([
                JankenEnv.Guu, JankenEnv.Choki,
                JankenEnv.Paa
            ])
        ]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return np.random.randint(0, 3, state_batches[0].shape), state_batches, {}
        return (state_batches[0] + 1) % 3, state_batches, {}


config = {
    "env": JankenEnv,
    "gamma": 0.9,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    "num_workers": 0,
    "num_envs_per_worker": 4,
    "train_batch_size": 200,
    "multiagent": {
        "policies_to_train": ["learned"],
        "policies": {
            "change_in_order": (ChangeInOrder, Discrete(3), Discrete(3),
                                {}),
            "learned": (None, Discrete(3), Discrete(3), {
                "framework": "torch",
            }),
        },
        "policy_mapping_fn":
            lambda agent_id: "learned" if agent_id == "player1" else "change_in_order",
    },
    "framework": "torch",
}
trainer_obj = ppo.PPOTrainer(config=config)
for _ in range(100):
    results = trainer_obj.train()
    print(pretty_print(results))



# # original gym.env code (not yet implemented)
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



# # existing MultiAgentEnv code
# config = {
#     "env": RockPaperScissors,
#     "gamma": 0.9,
#     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#     "num_gpus": 0,
#     "num_workers": 0,
#     "num_envs_per_worker": 4,
#     "train_batch_size": 200,
#     "multiagent": {
#         "policies_to_train": ["learned"],
#         "policies": {
#             "always_same": (AlwaysSameHeuristic, Discrete(3), Discrete(3),
#                             {}),
#             "learned": (None, Discrete(3), Discrete(3), {
#                 "framework": "torch",
#             }),
#         },
#         "policy_mapping_fn":
#              lambda agent_id: "learned" if agent_id == "player1" else "always_same",
#     },
#     "framework": "torch",
# }
# trainer_obj = ppo.PPOTrainer(config=config)
# for _ in range(100):
#     results = trainer_obj.train()
#     print(pretty_print(result))



# # existing gym.Env code
# config = ppo.DEFAULT_CONFIG.copy()
# config["num_gpus"] = 0
# config["num_workers"] = 1
# trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
#
# for i in range(100):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#
#    if i % 10 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)

# env = gym.make("CartPole-v0")
# observation = env.reset()
#
# for i in range(10):
#     print(env.step(trainer.compute_action(observation)))
#     # env.render()
