from typing import Optional

from mjx.const import TileType


class Tile:
    """Tile.

    m1(0) = 0, ..., rd(3) = 135
    """

    def __init__(self, tile_id: int) -> None:
        assert 0 <= tile_id <= 135
        self._id = tile_id

    def __str__(self) -> str:
        return f"{self.type().name.lower()}({self._id % 4})"

    def id(self) -> int:
        return self._id

    def type(self) -> TileType:
        return TileType(self._id // 4)

    def is_red(self) -> bool:
        # m5, p5, s5
        return self._id in [16, 52, 88]

    def num(self) -> Optional[int]:
        if self._id >= 108:  # ew(0)
            return None
        return (self._id // 4) % 9 + 1
