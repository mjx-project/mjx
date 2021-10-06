from __future__ import annotations

import json
from typing import Optional

import _mjx  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.visualizer.svg import save_svg


class State:
    def __init__(self, state_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.State] = None  # type: ignore
        if state_json is None:
            return

        self._cpp_obj = _mjx.State(state_json)  # type: ignore

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()  # type: ignore

    def to_proto(self) -> mjxproto.State:
        json_data = self.to_json()
        return json_format.ParseDict(json.loads(json_data), mjxproto.State())  # type: ignore

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
