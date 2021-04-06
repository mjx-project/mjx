to_char = {
    0: "一",
    1: "二",
    2: "三",
    3: "四",
    4: "五",
    5: "六",
    6: "七",
    7: "八",
    8: "九",
    9: "①",
    10: "②",
    11: "③",
    12: "④",
    13: "⑤",
    14: "⑥",
    15: "⑦",
    16: "⑧",
    17: "⑨",
    18: "１",
    19: "２",
    20: "３",
    21: "４",
    22: "５",
    23: "６",
    24: "７",
    25: "８",
    26: "９",
    27: "東",
    28: "南",
    29: "西",
    30: "北",
    31: "白",
    32: "發",
    33: "中",
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


def get_tile_char(tile_id: int) -> str:
    if 0 <= tile_id < 136:
        return to_char[tile_id // 4]
    else:
        return " "


def get_tile_unicode(tile_id: int) -> str:
    if 0 <= tile_id < 136:
        return to_unicode[tile_id // 4]
    else:
        return " "


def get_wind_char(wind: int) -> str:
    if 0 <= wind < 4:
        return to_wind_char[wind]
    else:
        return " "
