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
    0: "東",
    1: "南",
    2: "西",
    3: "北",
}
to_actiontype = {
    0: "ACTION_TYPE_DISCARD",
    1: "ACTION_TYPE_RIICHI",
    2: "ACTION_TYPE_TSUMO",
    3: "ACTION_TYPE_KAN_CLOSED",
    4: "ACTION_TYPE_KAN_ADDED",
    5: "ACTION_TYPE_KYUSYU",
    6: "ACTION_TYPE_NO",
    7: "ACTION_TYPE_CHI",
    8: "ACTION_TYPE_PON",
    9: "ACTION_TYPE_KAN_OPENED",
    10: "ACTION_TYPE_RON",
}
to_modifier = {
    0: "",
    1: "_R",  # Right
    2: "_M",  # Mid
    3: "_L",  # Left
    4: "_S",  # Self(kan closed)
    5: "_R(Add)",  # Right(kan added)
    6: "_M(Add)",  # Mid(kan added)
    7: "_L(Add)",  # Left(kan added)
    8: "*",  # TSUMOGIRI
}


def get_tile_char(tile_id: int, is_using_unicode: bool) -> str:
    if tile_id < 0 or 132 < tile_id:
        return " "
    if is_using_unicode:
        return to_unicode[tile_id // 4]
    else:
        return to_char[tile_id // 4]


def get_wind_char(wind: int) -> str:
    if 0 <= wind < 4:
        return to_wind_char[wind]
    else:
        return " "


def get_actiontype(action: int) -> str:
    if 0 <= action < 11:
        return to_actiontype[action]
    else:
        return " "


def get_modifier(modifier_id: int) -> str:
    if 0 <= modifier_id < 9:
        return to_modifier[modifier_id]
    else:
        return " "
