from __future__ import annotations

from typing import List

from google.protobuf import json_format

import mjxproto
import _mjx  # type: ignore

from mjx.open import Open
from mjx.tile import Tile

class Hand:

    def closed_tiles(self) -> List[Tile]:
        assert self._cpp_obj is not None
        return [Tile(tid) for tid in self.to_proto().closed_tiles]  # type: ignore

    def opens(self) -> List[Open]:
        assert self._cpp_obj is not None
        return [Open(bit) for bit in self.to_proto().opens]  # type: ignore

    def shanten_number(self) -> int:
        closed_tiles = self.closed_tiles()
        tile_type_cnt: List[int] = [0] * 34
        for tile in closed_tiles:
            tile_type_cnt[tile.type()] += 1

        return _mjx.ShantenCalculator.shanten_number(tile_type_cnt, len(self.opens()))

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()  # type: ignore

    def to_proto(self) -> mjxproto.Hand:
        assert self._cpp_obj is not None
        return json_format.Parse(self.to_json(), mjxproto.Hand())


    @classmethod
    def _from_cpp_obj(cls, cpp_obj: _mjx.Hand) -> Hand:  # type: ignore
        hand = cls()
        hand._cpp_obj = cpp_obj
        return hand
