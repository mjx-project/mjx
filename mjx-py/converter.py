from enum import Enum


class TileUnitType(Enum):
    HAND = 0
    CHI = 1
    PON = 2
    CLOSED_KAN = 3
    OPEN_KAN = 4
    ADDED_KAN = 5
    DISCARD = 6


class FromWho(Enum):
    NONE = 0
    RIGHT = 1
    MID = 2
    LEFT = 3
    SELF = 4


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
to_modifier = {
    FromWho.NONE: "",
    FromWho.RIGHT: "R",  # Right
    FromWho.MID: "M",  # Mid
    FromWho.LEFT: "L",  # Left
    FromWho.SELF: "S",  # Self(kan closed)
}
to_modifier_add_kan = {
    FromWho.RIGHT: "R(Add)",  # Right(kan added)
    FromWho.MID: "M(Add)",  # Mid(kan added)
    FromWho.LEFT: "L(Add)",  # Left(kan added)
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
end_type = {
    6: "ABORTIVE_DRAW_NINE_TERMINALS",
    12: "DRAW",
    13: "RIICHI_SCORE_CHANGE",
    14: "NEW_DORA",
    15: "ABORTIVE_DRAW_FOUR_RIICHIS",
    16: "ABORTIVE_DRAW_THREE_RONS",
    17: "ABORTIVE_DRAW_FOUR_KANS",
    18: "ABORTIVE_DRAW_FOUR_WINDS",
    19: "EXHAUSTIVE_DRAW_NORMAL",
    20: "EXHAUSTIVE_DRAW_NAGASHI_MANGAN",
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


def get_modifier(from_who: FromWho, tile_unit_type: TileUnitType) -> str:
    if tile_unit_type == TileUnitType.ADDED_KAN:
        return to_modifier_add_kan[from_who]
    else:
        return to_modifier[from_who]


def get_yaku(yaku: int) -> str:
    return yaku_list[yaku]


def get_end_type(last_event: int) -> str:
    return end_type[last_event]
