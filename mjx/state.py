from __future__ import annotations

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, State):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()  # type: ignore

    def to_proto(self) -> mjxproto.State:
        assert self._cpp_obj is not None
        return json_format.Parse(self.to_json(), mjxproto.State())

    def save_svg(self, filename: str, view_idx: int = 0):
        assert filename.endswith(".svg")
        assert 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)

    @classmethod
    def from_proto(cls, proto: mjxproto.State) -> State:
        return State(json_format.MessageToJson(proto))

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> State:
        state = cls()
        state._cpp_obj = cpp_obj
        return state
