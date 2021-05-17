from typing import List

from mjconvert.open_tile_ids_converter import open_tile_ids


class Hand:
    def __init__(self, closed_tiles: List[int] = [], opens: List[int] = []):
        self._closed_tiles: List[int] = closed_tiles  # each item is tile id (0 ~ 135)
        self.opens: List[int] = opens  # e.g., 49495. Follow Tenhou format.
        self.closed_tiles.sort()

    @property
    def closed_tiles(self):
        self._closed_tiles.sort()
        return self._closed_tiles

    def add(self, tile_id: int):
        """Used for draw and ron"""
        assert len(self._closed_tiles) in [1, 4, 7, 10, 13]
        assert tile_id not in self._closed_tiles
        self.closed_tiles.append(tile_id)

    def discard(self, tile_id: int):
        assert len(self._closed_tiles) in [2, 5, 8, 11, 14]
        assert tile_id in self._closed_tiles
        self.closed_tiles.remove(tile_id)

    def apply_open(self, open: int):
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
        # convert into tile_ids
        ids = open_tile_ids(open)
        for id in ids:
            if id not in self.closed_tiles:
                continue
            self.closed_tiles.remove(id)

        self.opens.append(open)
