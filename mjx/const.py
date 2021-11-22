from enum import IntEnum

import mjxproto


class ActionType(IntEnum):
    # After draw
    DISCARD = mjxproto.ACTION_TYPE_DISCARD
    TSUMOGIRI = mjxproto.ACTION_TYPE_TSUMOGIRI
    RIICHI = mjxproto.ACTION_TYPE_RIICHI
    CLOSED_KAN = mjxproto.ACTION_TYPE_CLOSED_KAN
    ADDED_KAN = mjxproto.ACTION_TYPE_ADDED_KAN
    TUSMO = mjxproto.ACTION_TYPE_TSUMO
    ABORTIVE_DRAW_NINE_TERMINALS = mjxproto.ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS  # 九種九牌
    # After other's discard
    CHI = mjxproto.ACTION_TYPE_CHI
    PON = mjxproto.ACTION_TYPE_PON
    OPEN_KAN = mjxproto.ACTION_TYPE_OPEN_KAN
    RON = mjxproto.ACTION_TYPE_RON
    PASS = mjxproto.ACTION_TYPE_NO
    # Dummy used only to check connection and share round terminal information
    DUMMY = mjxproto.ACTION_TYPE_DUMMY


class AbsolutePos(IntEnum):
    INIT_EAST = 0
    INIT_SOUTH = 1
    INIT_WEST = 2
    INIT_NORTH = 3


class RelativePos(IntEnum):
    SELF = 0
    RIGHT = 1
    MID = 2
    LEFT = 3


class SeedType(IntEnum):
    RANDOM = 0
    DUPLICATE = 1


class TileType(IntEnum):
    M1 = 0
    M2 = 1
    M3 = 2
    M4 = 3
    M5 = 4
    M6 = 5
    M7 = 6
    M8 = 7
    M9 = 8
    P1 = 9
    P2 = 10
    P3 = 11
    P4 = 12
    P5 = 13
    P6 = 14
    P7 = 15
    P8 = 16
    P9 = 17
    S1 = 18
    S2 = 19
    S3 = 20
    S4 = 21
    S5 = 22
    S6 = 23
    S7 = 24
    S8 = 25
    S9 = 26
    EW = 27
    SW = 28
    WW = 29
    NW = 30
    WD = 31
    GD = 32
    RD = 33
