from mjconvert import mjlog_decoder


def test_parse_wall():
    wall_outputs = [
        "124,122,37,27,80,127,125,87,104,67,115,95,8,57,92,130,69,118,20,128,35,6,123,56,103,96,55,85,109,88,32,63,26,117,16,17,82,47,68,23,9,25,65,5,39,94,76,58,97,36,14,99,111,7,133,113,31,100,131,70,28,46,30,60,79,41,74,116,75,93,105,49,91,135,114,42,45,132,21,119,18,24,129,51,121,3,81,40,29,13,34,19,86,78,53,64,50,71,120,83,66,84,126,4,12,10,101,102,22,112,15,48,134,77,0,11,108,98,61,1,107,110,59,90,2,44,54,38,89,33,62,43,73,106,72,52",
        "4,5",
    ]
    wall_dices = mjlog_decoder.parse_wall(wall_outputs)
    assert len(wall_dices) == 1
    wall, dice = wall_dices[0]
    assert wall == [
        52,
        72,
        106,
        73,
        43,
        62,
        33,
        89,
        38,
        54,
        44,
        2,
        90,
        59,
        110,
        107,
        1,
        61,
        98,
        108,
        11,
        0,
        77,
        134,
        48,
        15,
        112,
        22,
        102,
        101,
        10,
        12,
        4,
        126,
        84,
        66,
        83,
        120,
        71,
        50,
        64,
        53,
        78,
        86,
        19,
        34,
        13,
        29,
        40,
        81,
        3,
        121,
        51,
        129,
        24,
        18,
        119,
        21,
        132,
        45,
        42,
        114,
        135,
        91,
        49,
        105,
        93,
        75,
        116,
        74,
        41,
        79,
        60,
        30,
        46,
        28,
        70,
        131,
        100,
        31,
        113,
        133,
        7,
        111,
        99,
        14,
        36,
        97,
        58,
        76,
        94,
        39,
        5,
        65,
        25,
        9,
        23,
        68,
        47,
        82,
        17,
        16,
        117,
        26,
        63,
        32,
        88,
        109,
        85,
        55,
        96,
        103,
        56,
        123,
        6,
        35,
        128,
        20,
        118,
        69,
        130,
        92,
        57,
        8,
        95,
        115,
        67,
        104,
        87,
        125,
        127,
        80,
        27,
        37,
        122,
        124,
    ]
    assert dice == [4, 5]
