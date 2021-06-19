from typing import List

from mjx.env import RLlibMahjongEnv
import mjxproto
import ray
import ray.rllib.agents.ppo as ppo


def feature_extraction(observation: mjxproto.Observation) -> List[int]:
    return []


def decode_action(action: int) -> mjxproto.Action:
    return mjxproto.Action()


ray.init()
agent = ppo.PPOTrainer(env=RLlibMahjongEnv,  config={
    "env_config": {},})