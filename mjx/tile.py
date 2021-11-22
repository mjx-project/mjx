from enum import IntEnum
from typing import Optional

from mjx.const import TileType


class Tile(IntEnum):
    M1_0, M1_1, M1_2, M1_3 = 0, 1, 2, 3
    M2_0, M2_1, M2_2, M2_3 = 4, 5, 6, 7
    M3_0, M3_1, M3_2, M3_3 = 8, 9, 10, 11
    M4_0, M4_1, M4_2, M4_3 = 12, 13, 14, 15
    M5_0, M5_1, M5_2, M5_3 = 16, 17, 18, 19
    M6_0, M6_1, M6_2, M6_3 = 20, 21, 22, 23
    M7_0, M7_1, M7_2, M7_3 = 24, 25, 26, 27
    M8_0, M8_1, M8_2, M8_3 = 28, 29, 30, 31
    M9_0, M9_1, M9_2, M9_3 = 32, 33, 34, 35
    P1_0, P1_1, P1_2, P1_3 = 36, 37, 38, 39
    P2_0, P2_1, P2_2, P2_3 = 40, 41, 42, 43
    P3_0, P3_1, P3_2, P3_3 = 44, 45, 46, 47
    P4_0, P4_1, P4_2, P4_3 = 48, 49, 50, 51
    P5_0, P5_1, P5_2, P5_3 = 52, 53, 54, 55
    P6_0, P6_1, P6_2, P6_3 = 56, 57, 58, 59
    P7_0, P7_1, P7_2, P7_3 = 60, 61, 62, 63
    P8_0, P8_1, P8_2, P8_3 = 64, 65, 66, 67
    P9_0, P9_1, P9_2, P9_3 = 68, 69, 70, 71
    S1_0, S1_1, S1_2, S1_3 = 72, 73, 74, 75
    S2_0, S2_1, S2_2, S2_3 = 76, 77, 78, 79
    S3_0, S3_1, S3_2, S3_3 = 80, 81, 82, 83
    S4_0, S4_1, S4_2, S4_3 = 84, 85, 86, 87
    S5_0, S5_1, S5_2, S5_3 = 88, 89, 90, 91
    S6_0, S6_1, S6_2, S6_3 = 92, 93, 94, 95
    S7_0, S7_1, S7_2, S7_3 = 96, 97, 98, 99
    S8_0, S8_1, S8_2, S8_3 = 100, 101, 102, 103
    S9_0, S9_1, S9_2, S9_3 = 104, 105, 106, 107
    EW_0, EW_1, EW_2, EW_3 = 108, 109, 110, 111
    SW_0, SW_1, SW_2, SW_3 = 112, 113, 114, 115
    WW_0, WW_1, WW_2, WW_3 = 116, 117, 118, 119
    NW_0, NW_1, NW_2, NW_3 = 120, 121, 122, 123
    WD_0, WD_1, WD_2, WD_3 = 124, 125, 126, 127
    GD_0, GD_1, GD_2, GD_3 = 128, 129, 130, 131
    RD_0, RD_1, RD_2, RD_3 = 132, 133, 134, 135
    M5_RED, P5_RED, S5_RED = 16, 52, 88

    def id(self) -> int:
        return self.value

    def type(self) -> TileType:
        return TileType(self.value // 4)

    def is_red(self) -> bool:
        return self.value in [Tile.M5_RED, Tile.P5_RED, Tile.S5_RED]

    def num(self) -> Optional[int]:
        if self.value >= Tile.EW_0:
            return None
        return (self.value // 4) % 9 + 1
