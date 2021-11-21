from mjx.const import TileType


class Tile:
    def __init__(self, tid):
        self.tid = tid

    def type(self) -> TileType:
        return TileType(self.tid // 4)

    def is_red(self) -> bool:
        # m5(red), p5(red), s5(red)
        return self.tid in [16, 52, 88]
