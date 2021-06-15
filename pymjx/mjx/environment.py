import _mjx
from mjx_pb2_grpc import *
from mjx_pb2 import *
from google.protobuf import json_format


class MahjongEnv():
    def __init__(self):
        self.json_env = _mjx.RLlibMahjongEnv()

    def step(self, actions: dict[str, Action]) -> (dict[str, Observation], dict[str, int], dict[str, bool], dict[str, str]):
        json_observation, rewards, dones, info = \
            self.json_env.step({id: json_format.MessageToJson(act, "") for id, act in actions.items()})
        return {id: json_format.Parse(obs, Observation()) for id, obs in json_observation.items()}, rewards, dones, info

    def reset(self) -> dict[str, Observation]:
        return {id: json_format.Parse(obs, Observation()) for id, obs in self.json_env.reset().items()}

    def seed(self, seed: int):
        self.json_env.seed(seed)
