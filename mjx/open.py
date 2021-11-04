from __future__ import annotations

from typing import List

import _mjx  # type: ignore


class Open:
    @classmethod
    def event_type(cls, bit: int) -> int:
        return _mjx.Open.event_type(bit)

    @classmethod
    def steal_from(cls, bit: int) -> int:
        return _mjx.Open.steal_from(bit)

    @classmethod
    def at(cls, bit: int, i: int) -> int:
        return _mjx.Open.at(bit, i)

    @classmethod
    def size(cls, bit: int) -> int:
        return _mjx.Open.size(bit)

    @classmethod
    def tiles(cls, bit: int) -> List[int]:
        return _mjx.Open.tiles(bit)

    @classmethod
    def tiles_from_hand(cls, bit: int) -> List[int]:
        return _mjx.Open.tiles_from_hand(bit)

    @classmethod
    def stolen_tile(cls, bit: int) -> int:
        return _mjx.Open.stolen_tile(bit)

    @classmethod
    def last_tile(cls, bit: int) -> int:
        return _mjx.Open.last_tile(bit)

    @classmethod
    def undiscardable_tile_types(cls, bit: int) -> List[int]:
        return _mjx.Open.undiscardable_tile_types(bit)
