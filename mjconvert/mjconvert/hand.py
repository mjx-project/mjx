from typing import List

from mjconvert.open_tile_ids_converter import open_event_type, open_tile_ids


class Hand:
    def __init__(self, closed_tiles: List[int] = [], open_tiles: List[int] = []):
        self.closed_tiles: List[int] = closed_tiles  # each item is tile id (0 ~ 135)
        self.open_tiles: List[int] = open_tiles
        self.closed_tiles.sort()

    def draw(self, tile_id: int):
        self.closed_tiles.append(tile_id)
        self.closed_tiles.sort()

    def discard(self, tile_id: int):
        self.closed_tiles.remove(tile_id)

    def apply_open(self, open: int):
        """
        各鳴きをHandに適用し、closed_tilesとopen_tilesを変更する
        >>> hand = Hand([2, 4, 6, 5, 11, 10, 8, 9, 86, 90, 125, 126], [0, 3, 1])
        >>> hand.apply_open(49495)  # chi
        >>> hand.closed_tiles
        [2, 4, 5, 6, 8, 9, 10, 11, 125, 126]
        >>> hand.open_tiles
        [0, 3, 1, 82, 86, 90]
        >>> hand.apply_open(47723)  # pon
        >>> hand.closed_tiles
        [2, 4, 5, 6, 8, 9, 10, 11]
        >>> hand.open_tiles
        [0, 3, 1, 82, 86, 90, 124, 125, 126]
        >>> hand.apply_open(1793)  # open_kan
        >>> hand.closed_tiles
        [2, 8, 9, 10, 11]
        >>> hand.open_tiles
        [0, 3, 1, 82, 86, 90, 124, 125, 126, 4, 5, 6, 7]
        >>> hand.apply_open(2048)  # closed_kan
        >>> hand.closed_tiles
        [2]
        >>> hand.open_tiles
        [0, 3, 1, 82, 86, 90, 124, 125, 126, 4, 5, 6, 7, 8, 9, 10, 11]
        >>> hand.apply_open(530)  # added_kan
        >>> hand.closed_tiles
        []
        >>> hand.open_tiles
        [0, 3, 1, 82, 86, 90, 124, 125, 126, 4, 5, 6, 7, 8, 9, 10, 11, 2]
        """
        # convert into t  ile_ids
        ids = open_tile_ids(open)
        # move the tiles from close to open
        for id in ids:
            if self.closed_tiles.count(id):
                self.closed_tiles.remove(id)
            if not self.open_tiles.count(id):
                self.open_tiles.append(id)
