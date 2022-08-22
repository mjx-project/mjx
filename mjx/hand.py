from __future__ import annotations

from typing import List, Optional

import _mjx  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.const import TileType
from mjx.open import Open
from mjx.tile import Tile


class Hand:
    def __init__(self, hand_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.Hand] = None  # type: ignore
        if hand_json is None:
            return

        self._cpp_obj = _mjx.Hand(hand_json)  # type: ignore

    def closed_tiles(self) -> List[Tile]:
        assert self._cpp_obj is not None  # type: ignore
        return [Tile(tid) for tid in self._cpp_obj.closed_tiles()]  # type: ignore

    def closed_tile_types(self) -> List[TileType]:
        assert self._cpp_obj is not None  # type: ignore
        return [TileType(tt) for tt in self._cpp_obj.closed_tile_types()]  # type: ignore

    def opens(self) -> List[Open]:
        assert self._cpp_obj is not None  # type: ignore
        return [Open(bit) for bit in self._cpp_obj.opens()]  # type: ignore

    def shanten_number(self) -> int:
        assert self._cpp_obj is not None  # type: ignore
        return self._cpp_obj.shanten_number()  # type: ignore

    def effective_draw_types(self) -> List[TileType]:
        "Retrun the list of tile types which reduce the shanten number by drawing it"
        return [TileType(i) for i in self._cpp_obj.effective_draw_types()]  # type: ignore

    def effective_discard_types(self) -> List[TileType]:
        "Retrun the list of tile types which reduce the shanten number by discarding it"
        return [TileType(i) for i in self._cpp_obj.effective_discard_types()]  # type: ignore

    def to_json(self) -> str:
        assert self._cpp_obj is not None  # type: ignore
        return self._cpp_obj.to_json()  # type: ignore

    def to_proto(self) -> mjxproto.Hand:
        assert self._cpp_obj is not None  # type: ignore
        return json_format.Parse(self.to_json(), mjxproto.Hand())

    @classmethod
    def _from_cpp_obj(cls, cpp_obj: _mjx.Hand) -> Hand:  # type: ignore
        hand = cls()
        hand._cpp_obj = cpp_obj  # type: ignore
        return hand
