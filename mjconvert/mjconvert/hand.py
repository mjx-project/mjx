from mjconvert.open_tile_ids_converter import open_additional_tile_ids, open_removable_tile_ids
from typing import List
import doctest

class Hand:
    def __init__(self, closed_tiles: List[int] = [], open_tiles: List[int] = []):
        self.closed_tiles_: List[int] = closed_tiles  # each item is tile id (0 ~ 135)
        self.opens_: List[int] = open_tiles
        self.closed_tiles_.sort()
        self.opens_.sort()

    def draw(self, tile_id: int):
        self.closed_tiles_.append(tile_id)
        self.closed_tiles_.sort()

    def discard(self, tile_id: int):
        self.remove(tile_id)

    def apply_open(self, open: int):
        """
        chi,
        pon,
        open_kan,
        closed_kan,
        added_kan
        >>> hand = Hand([2, 4, 6, 5, 11, 10, 8, 9, 86, 90, 125, 126], [0, 3, 1])
        >>> hand.apply_open(49495)
        >>> hand.closed_tiles_
        [2, 4, 5, 6, 8, 9, 10, 11, 125, 126]
        >>> hand.opens_
        [0, 1, 3, 82, 86, 90]
        >>> hand.apply_open(47723)
        >>> hand.closed_tiles_
        [2, 4, 5, 6, 8, 9, 10, 11]
        >>> hand.opens_
        [0, 1, 3, 82, 86, 90, 124, 125, 126]
        >>> hand.apply_open(1793)
        >>> hand.closed_tiles_
        [2, 8, 9, 10, 11]
        >>> hand.opens_
        [0, 1, 3, 4, 5, 6, 7, 82, 86, 90, 124, 125, 126]
        >>> hand.apply_open(2048)
        >>> hand.closed_tiles_
        [2]
        >>> hand.opens_
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 82, 86, 90, 124, 125, 126]
        >>> hand.apply_open(530)
        >>> hand.closed_tiles_
        []
        >>> hand.opens_
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 82, 86, 90, 124, 125, 126]
        """
        # convert into t  ile_ids
        addtional_ids = open_additional_tile_ids(open)
        removable_ids = open_removable_tile_ids(open)
        # move the tiles from close to open
        # remove from close
        for id in removable_ids:
            self.closed_tiles_.remove(id)
        # add to open
        for id in addtional_ids:
            self.opens_.append(id)

        self.opens_.sort()


