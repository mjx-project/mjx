from enum import IntEnum


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
