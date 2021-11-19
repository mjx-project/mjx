from __future__ import annotations

from typing import List

import _mjx  # type: ignore


class Open:
    def __init__(self, bit: int):
        self.bit = bit

    def event_type(self) -> int:
        return _mjx.Open.event_type(self.bit)

    def steal_from(self) -> int:
        return _mjx.Open.steal_from(self.bit)

    def at(self, i: int) -> int:
        return _mjx.Open.at(self.bit, i)

    def size(self) -> int:
        return _mjx.Open.size(self.bit)

    def tiles(self) -> List[int]:
        return _mjx.Open.tiles(self.bit)

    def tiles_from_hand(self) -> List[int]:
        return _mjx.Open.tiles_from_hand(self.bit)

    def stolen_tile(self) -> int:
        return _mjx.Open.stolen_tile(self.bit)

    def last_tile(self) -> int:
        return _mjx.Open.last_tile(self.bit)

    def undiscardable_tile_types(self) -> List[int]:
        return _mjx.Open.undiscardable_tile_types(self.bit)
