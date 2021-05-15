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


class EventType(IntEnum):
    ABORTIVE_DRAW_NINE_TERMINALS = 6
    ABORTIVE_DRAW_FOUR_RIICHIS = 15
    ABORTIVE_DRAW_THREE_RONS = 16
    ABORTIVE_DRAW_FOUR_KANS = 17
    ABORTIVE_DRAW_FOUR_WINDS = 18
    EXHAUSTIVE_DRAW_NORMAL = 19
    EXHAUSTIVE_DRAW_NAGASHI_MANGAN = 20
