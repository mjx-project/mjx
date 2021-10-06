from __future__ import annotations

from typing import List, Optional

import _mjx  # type: ignore
import numpy as np  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.action import Action
from mjx.visualizer.svg import save_svg


class Observation:
    def __init__(self, obs_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.Observation] = None  # type: ignore
        if obs_json is None:
            return

        self._cpp_obj = _mjx.Observation(obs_json)  # type: ignore

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def legal_actions(self) -> List[Action]:
        assert self._cpp_obj is not None
        return [Action._from_cpp_obj(cpp_obj) for cpp_obj in self._cpp_obj.legal_actions()]  # type: ignore

    def action_mask(self, dtype=np.float32) -> np.ndarray:
        assert self._cpp_obj is not None
        return np.array(self._cpp_obj.action_mask(), dtype=dtype)  # type: ignore

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()  # type: ignore

    def to_proto(self) -> mjxproto.Observation:
        assert self._cpp_obj is not None
        return json_format.Parse(self.to_json(), mjxproto.Observation())

    def save_svg(self, filename: str, view_idx: Optional[int] = None) -> None:
        assert filename.endswith(".svg")
        assert view_idx is None or 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)

    @classmethod
    def from_proto(cls, proto: mjxproto.Observation) -> Observation:
        return Observation(json_format.MessageToJson(proto))

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> Observation:
        obs = cls()
        obs._cpp_obj = cpp_obj
        return obs
