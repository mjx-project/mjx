from __future__ import annotations

from typing import List, Optional, Tuple

import _mjx  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.action import Action
from mjx.observation import Observation
from mjx.visualizer.svg import save_svg, show_svg, to_svg


class State:
    def __init__(self, state_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.State] = None  # type: ignore
        if state_json is None:
            return

        self._cpp_obj = _mjx.State(state_json)  # type: ignore

    def _repr_html_(self) -> None:
        observation = self.to_proto()
        return to_svg(observation, target_idx=None)

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

    def past_decisions(self) -> List[Tuple[Observation, Action]]:
        return [
            (Observation._from_cpp_obj(obs), Action._from_cpp_obj(act))
            for obs, act in self._cpp_obj.past_decisions()
        ]

    def save_svg(self, filename: str, view_idx: int = 0):
        assert filename.endswith(".svg")
        assert 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)

    def show_svg(self, view_idx: Optional[int] = None) -> None:
        assert view_idx is None or 0 <= view_idx < 4

        observation = self.to_proto()
        show_svg(observation, target_idx=view_idx)

    @classmethod
    def from_proto(cls, proto: mjxproto.State) -> State:
        return State(json_format.MessageToJson(proto))

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> State:
        state = cls()
        state._cpp_obj = cpp_obj
        return state
