from __future__ import annotations

from typing import Optional

from mjx.const import EventType, RelativePlayerIdx
from mjxproto import ActionType

to_char = [
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    "m8",
    "m9",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "ew",
    "sw",
    "ww",
    "nw",
    "wd",
    "gd",
    "rd",
]

to_unicode = [
    "\U0001F007",
    "\U0001F008",
    "\U0001F009",
    "\U0001F00A",
    "\U0001F00B",
    "\U0001F00C",
    "\U0001F00D",
    "\U0001F00E",
    "\U0001F00F",
    "\U0001F019",
    "\U0001F01A",
    "\U0001F01B",
    "\U0001F01C",
    "\U0001F01D",
    "\U0001F01E",
    "\U0001F01F",
    "\U0001F020",
    "\U0001F021",
    "\U0001F010",
    "\U0001F011",
    "\U0001F012",
    "\U0001F013",
    "\U0001F014",
    "\U0001F015",
    "\U0001F016",
    "\U0001F017",
    "\U0001F018",
    "\U0001F000",
    "\U0001F001",
    "\U0001F002",
    "\U0001F003",
    "\U0001F006",
    "\U0001F005",
    "\U0001F004\uFE0E",
]

to_wind_char = [
    "EAST",
    "SOUTH",
    "WEST",
    "NORTH",
    "東",
    "南",
    "西",
    "北",
]

to_relative_player_idx = {
    RelativePlayerIdx.RIGHT: "R",  # Right
    RelativePlayerIdx.CENTER: "C",  # Center
    RelativePlayerIdx.LEFT: "L",  # Left
    RelativePlayerIdx.SELF: "S",  # Self(kan closed)
}

yaku_list = [
    "門前清自摸和",
    "立直",
    "一発",
    "槍槓",
    "嶺上開花",
    "海底摸月",
    "河底撈魚",
    "平和",
    "断幺九",
    "一盃口",
    "自風 東",
    "自風 南",
    "自風 西",
    "自風 北",
    "場風 東",
    "場風 南",
    "場風 西",
    "場風 北",
    "役牌 白",
    "役牌 發",
    "役牌 中",
    "両立直",
    "七対子",
    "混全帯幺九",
    "一気通貫",
    "三色同順",
    "三色同刻",
    "三槓子",
    "対々和",
    "三暗刻",
    "小三元",
    "混老頭",
    "二盃口",
    "純全帯幺九",
    "混一色",
    "清一色",
    "人和",  # 天鳳は人和なし
    "天和(役満)",
    "地和(役満)",
    "大三元(役満)",
    "四暗刻(役満)",
    "四暗刻単騎(役満)",
    "字一色(役満)",
    "緑一色(役満)",
    "清老頭(役満)",
    "九蓮宝燈(役満)",
    "純正九蓮宝燈(役満)",
    "国士無双(役満)",
    "国士無双１３面(役満)",
    "大四喜(役満)",
    "小四喜(役満)",
    "四槓子(役満)",
    "ドラ",
    "裏ドラ",
    "赤ドラ",
]

event_type_en = {
    EventType.DISCARD: "DISCARD",
    EventType.TSUMOGIRI: "TSUMOGIRI",
    EventType.RIICHI: "RIICHI",
    EventType.CLOSED_KAN: "CLOSED_KAN",
    EventType.ADDED_KAN: "ADDED_KAN",
    EventType.TSUMO: "TSUMO",
    EventType.ABORTIVE_DRAW_NINE_TERMINALS: "ABORTIVE_DRAW_NINE_TERMINALS",
    EventType.CHI: "CHI",
    EventType.PON: "PON",
    EventType.OPEN_KAN: "OPEN_KAN",
    EventType.RON: "RON",
    EventType.DRAW: "DRAW",
    EventType.RIICHI_SCORE_CHANGE: "RIICHI_SCORE_CHANGE",
    EventType.NEW_DORA: "NEW_DORA",
    EventType.ABORTIVE_DRAW_FOUR_RIICHIS: "ABORTIVE_DRAW_FOUR_RIICHIS",
    EventType.ABORTIVE_DRAW_THREE_RONS: "ABORTIVE_DRAW_THREE_RONS",
    EventType.ABORTIVE_DRAW_FOUR_KANS: "ABORTIVE_DRAW_FOUR_KANS",
    EventType.ABORTIVE_DRAW_FOUR_WINDS: "ABORTIVE_DRAW_FOUR_WINDS",
    EventType.ABORTIVE_DRAW_NORMAL: "ABORTIVE_DRAW_NORMAL",
    EventType.ABORTIVE_DRAW_NAGASHI_MANGAN: "ABORTIVE_DRAW_NAGASHI_MANGAN",
}

event_type_ja = {
    EventType.DISCARD: "打牌",
    EventType.TSUMOGIRI: "ツモ切り",
    EventType.RIICHI: "リーチ",
    EventType.CLOSED_KAN: "暗槓",
    EventType.ADDED_KAN: "加槓",
    EventType.TSUMO: "ツモ",
    EventType.ABORTIVE_DRAW_NINE_TERMINALS: "九種九牌",
    EventType.CHI: "チー",
    EventType.PON: "ポン",
    EventType.OPEN_KAN: "大明槓",
    EventType.RON: "ロン",
    EventType.DRAW: "引き分け",
    EventType.RIICHI_SCORE_CHANGE: "リーチ（スコア変動）",
    EventType.NEW_DORA: "新ドラ",
    EventType.ABORTIVE_DRAW_FOUR_RIICHIS: "四家立直",
    EventType.ABORTIVE_DRAW_THREE_RONS: "三家和了",
    EventType.ABORTIVE_DRAW_FOUR_KANS: "四槓散了",
    EventType.ABORTIVE_DRAW_FOUR_WINDS: "四風連打",
    EventType.ABORTIVE_DRAW_NORMAL: "流局",
    EventType.ABORTIVE_DRAW_NAGASHI_MANGAN: "流し満貫",
}

action_type_en = {
    ActionType.ACTION_TYPE_DISCARD: "DISCARD",
    ActionType.ACTION_TYPE_TSUMOGIRI: "TSUMOGIRI",
    ActionType.ACTION_TYPE_RIICHI: "RIICHI",
    ActionType.ACTION_TYPE_CLOSED_KAN: "CLOSED_KAN",
    ActionType.ACTION_TYPE_ADDED_KAN: "ADDED_KAN",
    ActionType.ACTION_TYPE_TSUMO: "TSUMO",
    ActionType.ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS: "ABORTIVE_DRAW_NINE_TERMINALS",
    ActionType.ACTION_TYPE_CHI: "CHI",
    ActionType.ACTION_TYPE_PON: "PON",
    ActionType.ACTION_TYPE_OPEN_KAN: "OPEN_KAN",
    ActionType.ACTION_TYPE_RON: "RON",
    ActionType.ACTION_TYPE_NO: "PASS",
    ActionType.ACTION_TYPE_DUMMY: "DUMMY",
}

action_type_ja = {
    ActionType.ACTION_TYPE_DISCARD: "打牌",
    ActionType.ACTION_TYPE_TSUMOGIRI: "ツモ切り",
    ActionType.ACTION_TYPE_RIICHI: "リーチ",
    ActionType.ACTION_TYPE_CLOSED_KAN: "暗槓",
    ActionType.ACTION_TYPE_ADDED_KAN: "加槓",
    ActionType.ACTION_TYPE_TSUMO: "ツモ",
    ActionType.ACTION_TYPE_ABORTIVE_DRAW_NINE_TERMINALS: "九種九牌",
    ActionType.ACTION_TYPE_CHI: "チー",
    ActionType.ACTION_TYPE_PON: "ポン",
    ActionType.ACTION_TYPE_OPEN_KAN: "明槓",
    ActionType.ACTION_TYPE_RON: "ロン",
    ActionType.ACTION_TYPE_NO: "パス",
    ActionType.ACTION_TYPE_DUMMY: "ダミー",
}


def get_tile_char(tile_id: int, is_using_unicode: bool) -> str:
    if tile_id < 0 or 136 < tile_id:
        return " "
    if is_using_unicode:
        return to_unicode[tile_id // 4]
    else:
        return to_char[tile_id // 4]


def get_wind_char(wind: int, lang: int = 0) -> str:
    if 0 <= wind < 4:
        if lang == 1:
            return to_wind_char[wind + 4]
        return to_wind_char[wind]
    else:
        return " "


def get_modifier(from_who: Optional[RelativePlayerIdx], tile_unit_type: EventType) -> str:
    if from_who is None:
        return ""
    if tile_unit_type == EventType.ADDED_KAN:
        return to_relative_player_idx[from_who] + "(Add)"
    else:
        return to_relative_player_idx[from_who]


def get_yaku(yaku: int) -> str:
    return yaku_list[yaku]


def get_event_type(last_event: EventType, lang: int) -> str:
    if lang == 0:
        return event_type_en[last_event]
    else:
        return event_type_ja[last_event]
