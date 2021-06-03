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


to_char = {
    0: "m1",
    1: "m2",
    2: "m3",
    3: "m4",
    4: "m5",
    5: "m6",
    6: "m7",
    7: "m8",
    8: "m9",
    9: "p1",
    10: "p2",
    11: "p3",
    12: "p4",
    13: "p5",
    14: "p6",
    15: "p7",
    16: "p8",
    17: "p9",
    18: "s1",
    19: "s2",
    20: "s3",
    21: "s4",
    22: "s5",
    23: "s6",
    24: "s7",
    25: "s8",
    26: "s9",
    27: "ew",
    28: "sw",
    29: "ww",
    30: "nw",
    31: "wd",
    32: "gd",
    33: "rd",
}
to_unicode = {
    0: "\U0001F007",
    1: "\U0001F008",
    2: "\U0001F009",
    3: "\U0001F00A",
    4: "\U0001F00B",
    5: "\U0001F00C",
    6: "\U0001F00D",
    7: "\U0001F00E",
    8: "\U0001F00F",
    9: "\U0001F019",
    10: "\U0001F01A",
    11: "\U0001F01B",
    12: "\U0001F01C",
    13: "\U0001F01D",
    14: "\U0001F01E",
    15: "\U0001F01F",
    16: "\U0001F020",
    17: "\U0001F021",
    18: "\U0001F010",
    19: "\U0001F011",
    20: "\U0001F012",
    21: "\U0001F013",
    22: "\U0001F014",
    23: "\U0001F015",
    24: "\U0001F016",
    25: "\U0001F017",
    26: "\U0001F018",
    27: "\U0001F000",
    28: "\U0001F001",
    29: "\U0001F002",
    30: "\U0001F003",
    31: "\U0001F006",
    32: "\U0001F005",
    33: "\U0001F004\uFE0E",
}
to_wind_char = {
    0: "EAST",
    1: "SOUTH",
    2: "WEST",
    3: "NORTH",
    4: "東",
    5: "南",
    6: "西",
    7: "北",
}
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
