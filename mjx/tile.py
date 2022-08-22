from typing import Optional

from mjx.const import TileType


class Tile:
    """Tile.

    Tile id (0 ~ 135):

        m1   0,   1,   2,   3
        m2   4,   5,   6,   7
        m3   8,   9,  10,  11
        m4  12,  13,  14,  15
        m5  16,  17,  18,  19
        m6  20,  21,  22,  23
        m7  24,  25,  26,  27
        m8  28,  29,  30,  31
        m9  32,  33,  34,  35
        p1  36,  37,  38,  39
        p2  40,  41,  42,  43
        p3  44,  45,  46,  47
        p4  48,  49,  50,  51
        p5  52,  53,  54,  55
        p6  56,  57,  58,  59
        p7  60,  61,  62,  63
        p8  64,  65,  66,  67
        p9  68,  69,  70,  71
        s1  72,  73,  74,  75
        s2  76,  77,  78,  79
        s3  80,  81,  82,  83
        s4  84,  85,  86,  87
        s5  88,  89,  90,  91
        s6  92,  93,  94,  95
        s7  96,  97,  98,  99
        s8 100, 101, 102, 103
        s9 104, 105, 106, 107
        ew 108, 109, 110, 111
        sw 112, 113, 114, 115
        ww 116, 117, 118, 119
        nw 120, 121, 122, 123
        wd 124, 125, 126, 127
        gd 128, 129, 130, 131
        rd 132, 133, 134, 135
    """

    def __init__(self, tile_id: int) -> None:
        assert 0 <= tile_id <= 135
        self._id = tile_id

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
