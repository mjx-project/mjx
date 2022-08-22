from __future__ import annotations

from typing import List, Optional

import _mjx  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.const import ActionType, PlayerIdx
from mjx.open import Open
from mjx.tile import Tile


class Action:
    def __init__(self, action_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.Action] = None  # type: ignore
        if action_json is None:
            return

        self._cpp_obj = _mjx.Action(action_json)  # type: ignore

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, Action)
        return self._cpp_obj == other._cpp_obj

    def __ne__(self, other: object) -> bool:
        assert isinstance(other, Action)
        return self._cpp_obj != other._cpp_obj

    def who(self) -> PlayerIdx:
        # TODO: remove to_proto()
        return PlayerIdx(self.to_proto().who)

    def open(self) -> Optional[Open]:
        o = self._cpp_obj._open()
        if o is None:
            return None
        return Open(o)  # type: ignore

    def type(self) -> ActionType:
        return ActionType(self._cpp_obj.type())  # type: ignore

    def tile(self) -> Optional[Tile]:
        t = self._cpp_obj.tile()  # type: ignore
        return Tile(t) if t is not None else None

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()

    def to_proto(self) -> mjxproto.Action:
        assert self._cpp_obj is not None
        return json_format.Parse(self.to_json(), mjxproto.Action())

    def to_idx(self) -> int:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_idx()

    @classmethod
    def select_from(cls, idx: int, legal_actions: List[Action]) -> Action:
        assert [idx in [action.to_idx() for action in legal_actions]]
        return Action._from_cpp_obj(_mjx.Action.select_from(idx, [a._cpp_obj for a in legal_actions]))  # type: ignore

    @classmethod
    def from_proto(cls, proto: mjxproto.Action) -> Action:
        return Action(json_format.MessageToJson(proto))

    @classmethod
    def _from_cpp_obj(cls, cpp_obj: _mjx.Action) -> Action:  # type: ignore
        action = cls()
        action._cpp_obj = cpp_obj
        return action
