from __future__ import annotations

from typing import List

import _mjx  # type: ignore

import mjxproto


class ShantenCalculator:
    @classmethod
    def shanten_number(cls, hand: mjxproto.Hand) -> int:
        closed_tiles = hand.closed_tiles
        tile_type_cnt: List[int] = [0] * 34
        for tile_id in closed_tiles:
            tile_type = tile_id // 4
            tile_type_cnt[tile_type] += 1

        return _mjx.ShantenCalculator.shanten_number(tile_type_cnt, len(hand.opens))

    @classmethod
    def proceeding_tile_types(cls, hand: mjxproto.Hand) -> int:
        closed_tiles = hand.closed_tiles
        tile_type_cnt: List[int] = [0] * 34
        for tile_id in closed_tiles:
            tile_type = tile_id // 4
            tile_type_cnt[tile_type] += 1

        return _mjx.ShantenCalculator.proceeding_tile_types(tile_type_cnt, len(hand.opens))
