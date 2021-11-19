
import _mjx  # type: ignore

from mjx.const import TileType

class Tile:
    def __init__(self, tid):
        self.tid = tid

    def type(self) -> TileType:
        return TileType(self.tid // 4)
