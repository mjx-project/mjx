from __future__ import annotations

from typing import Optional

import _mjx  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.const import EventType, PlayerIdx
from mjx.open import Open
from mjx.tile import Tile


class Event:
    def __init__(self, action_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.Event] = None  # type: ignore
        if action_json is None:
            return

        self._cpp_obj = _mjx.Event(action_json)  # type: ignore

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Event):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def open(self) -> Optional[Open]:
        o = self._cpp_obj._open()
        if o is None:
            return None
        return Open(o)  # type: ignore

    def type(self) -> EventType:
        return EventType(self._cpp_obj.type())  # type: ignore

    def who(self) -> PlayerIdx:
        return self._cpp_obj.who()  # type: ignore

    def tile(self) -> Optional[Tile]:
        t = self._cpp_obj.tile()  # type: ignore
        return Tile(t) if t is not None else None

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()

    def to_proto(self) -> mjxproto.Event:
        assert self._cpp_obj is not None
        return json_format.Parse(self.to_json(), mjxproto.Event())

    @classmethod
    def from_proto(cls, proto: mjxproto.Event) -> Event:
        return Event(json_format.MessageToJson(proto))

    @classmethod
    def _from_cpp_obj(cls, cpp_obj: _mjx.Event) -> Event:  # type: ignore
        event = cls()
        event._cpp_obj = cpp_obj
        return event
