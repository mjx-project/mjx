from enum import IntEnum

import mjxproto


class ActionType(IntEnum):
    # After draw
    DISCARD = mjxproto.ACTION_TYPE_DISCARD
    TSUMOGIRI = mjxproto.ACTION_TYPE_TSUMOGIRI
    RIICHI = mjxproto.ACTION_TYPE_RIICHI
    CLOSED_KAN = mjxproto.ACTION_TYPE_CLOSED_KAN
    ADDED_KAN = mjxproto.ACTION_TYPE_ADDED_KAN
    TSUMO = mjxproto.ACTION_TYPE_TSUMO
    ABORTIVE_DRAW_NINE_TERMINALS = mjxproto.ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS  # 九種九牌
    # After other's discard
    CHI = mjxproto.ACTION_TYPE_CHI
    PON = mjxproto.ACTION_TYPE_PON
    OPEN_KAN = mjxproto.ACTION_TYPE_OPEN_KAN
    RON = mjxproto.ACTION_TYPE_RON
    PASS = mjxproto.ACTION_TYPE_NO
    # Dummy used only to check connection and share round terminal information
    DUMMY = mjxproto.ACTION_TYPE_DUMMY


class EventType(IntEnum):
    # Publicly observable actions
    DISCARD = mjxproto.EVENT_TYPE_DISCARD
    TSUMOGIRI = mjxproto.EVENT_TYPE_TSUMOGIRI  # ツモ切り, Tsumogiri
    RIICHI = mjxproto.EVENT_TYPE_RIICHI
    CLOSED_KAN = mjxproto.EVENT_TYPE_CLOSED_KAN
    ADDED_KAN = mjxproto.EVENT_TYPE_ADDED_KAN
    TSUMO = mjxproto.EVENT_TYPE_TSUMO
    ABORTIVE_DRAW_NINE_TERMINALS = mjxproto.EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS
    CHI = mjxproto.EVENT_TYPE_CHI
    PON = mjxproto.EVENT_TYPE_PON
    OPEN_KAN = mjxproto.EVENT_TYPE_OPEN_KAN
    RON = mjxproto.EVENT_TYPE_RON
    # State transitions made by environment. There is no decision making by players.
    # 11 is skipped for the consistency to ActionType
    DRAW = mjxproto.EVENT_TYPE_DRAW
    RIICHI_SCORE_CHANGE = mjxproto.EVENT_TYPE_RIICHI_SCORE_CHANGE
    NEW_DORA = mjxproto.EVENT_TYPE_NEW_DORA
    ABORTIVE_DRAW_FOUR_RIICHIS = mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS  # 四家立直
    ABORTIVE_DRAW_THREE_RONS = mjxproto.EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS  # 三家和了
    ABORTIVE_DRAW_FOUR_KANS = mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS  # 四槓散了
    ABORTIVE_DRAW_FOUR_WINDS = mjxproto.EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS  # 四風連打
    ABORTIVE_DRAW_NORMAL = mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL  # 通常流局
    ABORTIVE_DRAW_NAGASHI_MANGAN = mjxproto.EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN  # 流し満貫


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


class PlayerIdx(IntEnum):
    INIT_EAST = 0  # Dealer of the 1st round, 起家
    INIT_SOUTH = 1
    INIT_WEST = 2
    INIT_NORTH = 3  # ラス親


class RelativePlayerIdx(IntEnum):
    SELF = 0  # 自家
    RIGHT = 1  # 下家
    CENTER = 2  # 対面
    LEFT = 3  # 上家
