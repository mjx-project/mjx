from typing import Dict, List

import mjx
import _mjx


class MjxEnv:
    def __init__(
        self,
        player_ids: List[str] = ["player_0", "player_1", "player_2", "player_3"],
    ):
        self._env = _mjx.MjxEnv(player_ids)

    def seed(self, seed) -> None:
        self._env.seed(seed)

    def reset(self) -> Dict[str, mjx.Observation]:
        cpp_obs_dict = self._env.reset()
        return {k: mjx.Observation._from_cpp_obj(v) for k, v in cpp_obs_dict.items()}

    def step(self, aciton_dict: Dict[str, mjx.Action]) -> Dict[str, mjx.Observation]:
        cpp_action_dict = {k: v._cpp_obj for k, v in aciton_dict.items()}
        cpp_obs_dict = self._env.step(cpp_action_dict)
        return {k: mjx.Observation._from_cpp_obj(v) for k, v in cpp_obs_dict.items()}

    def done(self) -> bool:
        return self._env.done()

    def rewards(self) -> Dict[str, int]:
        return self._env.rewards()

    @property
    def state(self) -> mjx.State:
        return mjx.State._from_cpp_obj(self._env.state())
