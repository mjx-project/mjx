from typing import Dict, Tuple

from google.protobuf import json_format

import mjxcore
import mjxproto


class RLlibMahjongEnv:
    def __init__(self):
        self.json_env = mjxcore.RLlibMahjongEnv()

    def step(
        self, actions: Dict[str, mjxproto.Action]
    ) -> Tuple[Dict[str, mjxproto.Observation], Dict[str, int], Dict[str, bool], Dict[str, str]]:
        json_observation, rewards, dones, info = self.json_env.step(
            {id: json_format.MessageToJson(act, False) for id, act in actions.items()}
        )
        return (
            {
                id: json_format.Parse(obs, mjxproto.Observation())
                for id, obs in json_observation.items()
            },
            rewards,
            dones,
            info,
        )

    def reset(self) -> Dict[str, mjxproto.Observation]:
        return {
            id: json_format.Parse(obs, mjxproto.Observation())
            for id, obs in self.json_env.reset().items()
        }

    def seed(self, seed: int):
        self.json_env.seed(seed)
