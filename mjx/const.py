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
