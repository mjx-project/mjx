from __future__ import annotations

from typing import List

import _mjx  # type: ignore

from mjx.const import EventType, RelativePlayerIdx, TileType
from mjx.tile import Tile


class Open:
    def __init__(self, bit: int):
        self.bit = bit

    def event_type(self) -> EventType:
        return _mjx.Open.event_type(self.bit)

    def steal_from(self) -> RelativePlayerIdx:
        return RelativePlayerIdx(_mjx.Open.steal_from(self.bit))

    def at(self, i: int) -> Tile:
        return Tile(_mjx.Open.at(self.bit, i))

    def size(self) -> int:
        return _mjx.Open.size(self.bit)

    def tiles(self) -> List[Tile]:
        return [Tile(t) for t in _mjx.Open.tiles(self.bit)]

    def tiles_from_hand(self) -> List[Tile]:
        return [Tile(t) for t in _mjx.Open.tiles_from_hand(self.bit)]

    def stolen_tile(self) -> Tile:
        return Tile(_mjx.Open.stolen_tile(self.bit))

    def last_tile(self) -> Tile:
        return Tile(_mjx.Open.last_tile(self.bit))

    def undiscardable_tile_types(self) -> List[TileType]:
        return [TileType(t) for t in _mjx.Open.undiscardable_tile_types(self.bit)]
