from typing import Dict, List

import _mjx  # type: ignore

from mjx.action import Action
from mjx.observation import Observation
from mjx.state import State


class MjxEnv:
    def __init__(
        self,
        player_ids: List[str] = ["player_0", "player_1", "player_2", "player_3"],
    ):
        self._env: _mjx.MjxEnv = _mjx.MjxEnv(player_ids)  # type: ignore

    def seed(self, seed) -> None:
        self._env.seed(seed)

    def reset(self) -> Dict[str, Observation]:
        cpp_obs_dict: Dict[str, _mjx.Observation] = self._env.reset()  # type: ignore
        return {k: Observation._from_cpp_obj(v) for k, v in cpp_obs_dict.items()}

    def step(self, aciton_dict: Dict[str, Action]) -> Dict[str, Observation]:
        cpp_action_dict: Dict[str, _mjx.Action] = {k: v._cpp_obj for k, v in aciton_dict.items()}  # type: ignore
        cpp_obs_dict: Dict[str, _mjx.Observation] = self._env.step(cpp_action_dict)  # type: ignore
        return {k: Observation._from_cpp_obj(v) for k, v in cpp_obs_dict.items()}

    def done(self) -> bool:
        return self._env.done()

    def rewards(self) -> Dict[str, int]:
        return self._env.rewards()

    @property
    def state(self) -> State:
        return State._from_cpp_obj(self._env.state())
