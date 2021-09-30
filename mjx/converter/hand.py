from typing import List

import mjxproto
from mjx.converter.open_converter import open_event_type
from mjx.converter.open_tile_ids_converter import open_tile_ids


class Hand:
    def __init__(self, closed_tiles: List[int] = [], opens: List[int] = []) -> None:
        self._closed_tiles: List[int] = closed_tiles  # each item is tile id (0 ~ 135)
        self.opens: List[int] = opens  # e.g., 49495. Follow Tenhou format.
        self.closed_tiles.sort()

    @property
    def closed_tiles(self) -> List[int]:
        self._closed_tiles.sort()
        return self._closed_tiles

    def add(self, tile_id: int) -> None:
        """Used for draw and ron"""
        assert len(self._closed_tiles) in [1, 4, 7, 10, 13]
        assert tile_id not in self._closed_tiles
        self.closed_tiles.append(tile_id)

    def discard(self, tile_id: int) -> None:
        assert len(self._closed_tiles) in [2, 5, 8, 11, 14]
        assert tile_id in self._closed_tiles
        self.closed_tiles.remove(tile_id)

    def apply_open(self, open: int) -> None:
        """
        Update closed_tiles and opens by open action.

        TODO: fix test cases

        >>> hand = Hand([2, 4, 6, 5, 11, 10, 8, 9, 86, 90, 125, 126, 130], [])
        >>> hand.apply_open(49495)  # chi
        >>> hand.closed_tiles
        [2, 4, 5, 6, 8, 9, 10, 11, 125, 126, 130]
        >>> hand.opens
        [49495]
        """
        if open_event_type(open) == mjxproto.EVENT_TYPE_ADDED_KAN:
            self._apply_added_kan(open)
            return

        ids = open_tile_ids(open)
        for id in ids:
            if id not in self.closed_tiles:
                continue
            self.closed_tiles.remove(id)

        self.opens.append(open)

        assert len(self.opens) == len(set(self.opens))

    def _apply_added_kan(self, open: int) -> None:
        assert open_event_type(open) == mjxproto.EVENT_TYPE_ADDED_KAN
        num_closed_tiles = len(self._closed_tiles)
        num_opens = len(self.opens)

        opens = self.opens
        for ix, old_open in enumerate(opens):
            if open_event_type(old_open) != mjxproto.EVENT_TYPE_PON:
                continue
            if open_tile_ids(old_open)[0] // 4 != open_tile_ids(open)[0] // 4:
                continue
            self.opens[ix] = open

        ids = open_tile_ids(open)
        for id in ids:
            if id not in self.closed_tiles:
                continue
            self.closed_tiles.remove(id)

        assert num_closed_tiles - 1 == len(self._closed_tiles)
        assert num_opens == len(self.opens)
        assert len(self.opens) == len(set(self.opens))
