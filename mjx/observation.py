from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
from google.protobuf import json_format

import mjx
import mjx._mjx as _mjx
import mjxproto
from mjx.visualizer.svg import save_svg


class Observation:
    def __init__(self, obs_json=None) -> None:
        self._cpp_obj = None
        if obs_json is None:
            return

        self._cpp_obj = _mjx.Observation(obs_json)

    def legal_actions(self) -> List[mjx.Action]:
        return [mjx.Action._from_cpp_obj(cpp_obj) for cpp_obj in self._cpp_obj.legal_actions()]

    def action_mask(self, dtype=np.float32) -> np.array:
        return np.array(self._cpp_obj.action_mask(), dtype=dtype)

    def to_json(self) -> str:
        return self._cpp_obj.to_json()

    def to_proto(self) -> mjxproto.Observation:
        json_data = self.to_json()
        return json_format.ParseDict(json.loads(json_data), mjxproto.Observation())

    def save_svg(self, filename: str, view_idx: Optional[int] = None) -> None:
        assert filename.endswith(".svg")
        assert view_idx is None or 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> Observation:
        obs = cls()
        obs._cpp_obj = cpp_obj
        return obs
