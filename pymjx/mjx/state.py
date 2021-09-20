from __future__ import annotations

import json

import mjx._mjx as _mjx
import mjxproto
from google.protobuf import json_format
from mjx.visualizer.svg import save_svg


class State:
    def __init__(self, state_json=None) -> None:
        self._cpp_obj = None
        if state_json is None:
            return

        self._cpp_obj = _mjx.State(state_json)

    def to_json(self) -> str:
        return self._cpp_obj.to_json()

    def to_proto(self) -> mjxproto.State:
        json_data = self.to_json()
        return json_format.ParseDict(json.loads(json_data), mjxproto.State())

    def save_svg(self, filename: str, view_idx: int = 0):
        assert filename.endswith(".svg")
        assert 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> State:
        state = cls()
        state._cpp_obj = cpp_obj
        return state
