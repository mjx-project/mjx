import base64
import os

import svgwrite
from mjx.visualizer.visualizer import (
    FromWho,
    MahjongTable,
    TileUnitType,
    get_tile_char,
    get_wind_char,
)


def dwg_add(dwg_p, dwg_g, pos, txt, rotate=False, transparent=False):
    opacity = 1.0
    if transparent:
        opacity = 0.5

    if rotate:
        if txt[1]:
            horizontal_tiles = [
                dwg_p.text(txt[0], insert=(0, 0), fill="red", opacity=opacity),
                dwg_p.text(
                    "\U0001F006",
                    insert=(0, 0),
                    stroke=svgwrite.rgb(255, 255, 255, "%"),
                    fill="white",
                ),
                dwg_p.text("\U0001F006", insert=(0, 0), fill="black", opacity=opacity),
            ]

            for horizontal_tile in horizontal_tiles:
                horizontal_tile.rotate(90, (0, 0))
                horizontal_tile.translate(pos)
                dwg_g.add(horizontal_tile)
        else:
            horizontal_tile = dwg_p.text(txt[0], insert=(0, 0), opacity=opacity)
            horizontal_tile.rotate(90, (0, 0))
            horizontal_tile.translate(pos)
            dwg_g.add(horizontal_tile)
    else:
        if txt[1]:
            dwg_g.add(dwg_p.text(txt[0], pos, fill="red", opacity=opacity))
            dwg_g.add(
                dwg_p.text(
                    "\U0001F006",
                    pos,
                    stroke=svgwrite.rgb(255, 255, 255, "%"),
                    fill="white",
                    opacity=opacity,
                )
            )
            dwg_g.add(dwg_p.text("\U0001F006", pos, fill="black", opacity=opacity))
        else:
            dwg_g.add(dwg_p.text(txt[0], pos, opacity=opacity))


def make_svg(filename: str, mode: str, page: int):
    width = 800
    height = 800
    char_width = 32  # 45:28.8,60:38.4
    char_height = 44  # 45:40.5,60:53
    red_hai = [16, 52, 88]

    data = MahjongTable.load_data(filename, mode)
    sample_data = data[page]
    sample_data.players.sort(key=lambda x: (x.player_idx - sample_data.my_idx) % 4)

    dwg = svgwrite.Drawing(
        filename.replace(".json", "") + "_" + str(page) + ".svg",
        (width, height),
        debug=True,
    )

    dwg.embed_font(
        name="GL-MahjongTile",
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "GL-MahjongTile.ttf"),
    )

    player_g = dwg.g()

    players = [[], [], [], []]
    pai = [[], [], [], []]
    player_info = [[], [], [], []]
    winds = [0, 0, 0, 0]
    scores = [0, 0, 0, 0]
    is_riichi = [False, False, False, False]

    hands = [[], [], [], []]
    open_tiles = [[], [], [], []]
    discards = [[], [], [], []]

    for i in range(4):  # iは各プレイヤー(0-3)
        players[i] = dwg.g()
        pai[i] = dwg.g(style="font-size:50;font-family:GL-MahjongTile;")
        player_info[i] = dwg.g()

        winds[i] = get_wind_char(sample_data.players[i].wind, lang=1)
        scores[i] = sample_data.players[i].score
        is_riichi[i] = sample_data.players[i].is_declared_riichi

        for t_u in reversed(sample_data.players[i].tile_units):
            if t_u.tile_unit_type == TileUnitType.HAND:
                for tile in t_u.tiles:
                    hands[i].append(
                        [
                            [
                                "\U0001F02B" if not tile.is_open else get_tile_char(tile.id, True),
                                tile.id in red_hai,
                            ]
                        ]
                    )

            if t_u.tile_unit_type == TileUnitType.DISCARD:
                for tile in t_u.tiles:
                    discards[i].append(
                        [
                            [
                                get_tile_char(tile.id, True),
                                tile.id in red_hai,
                            ],
                            tile.with_riichi,
                            tile.is_tsumogiri,
                            tile.is_transparent,
                        ]
                    )
            if t_u.tile_unit_type in [
                TileUnitType.CHI,
                TileUnitType.PON,
                TileUnitType.CLOSED_KAN,
                TileUnitType.OPEN_KAN,
                TileUnitType.ADDED_KAN,
            ]:
                open_tiles[i].append(
                    [
                        [[get_tile_char(tile.id, True), tile.id in red_hai] for tile in t_u.tiles],
                        t_u.from_who,
                        t_u.tile_unit_type,
                    ]
                )

    dwg.add(dwg.rect(insert=(278, 278), size=(244, 244)))
    dwg.add(dwg.rect(insert=(279, 279), size=(242, 242), fill="rgb(255,255,255)"))

    # board_info
    if sample_data.honba > 0:
        round = (
            get_wind_char((sample_data.round - 1) // 4, lang=1)
            + str((sample_data.round - 1) % 4 + 1)
            + "局"
        )
        dwg.add(dwg.text(round, (340, 360), style="font-size:24;font-family:serif;"))
        honba = str(sample_data.honba) + "本場"
        dwg.add(dwg.text(honba, (400, 360), style="font-size:24;font-family:serif;"))
    else:
        round = (
            get_wind_char((sample_data.round - 1) // 4, lang=1)
            + str((sample_data.round - 1) % 4 + 1)
            + "局"
        )
        dwg.add(dwg.text(round, (368, 360), style="font-size:26;font-family:serif;"))

    # dora
    doras = [get_tile_char(tile, True) for tile in sample_data.doras]
    while len(doras) < 5:
        doras.append("\U0001F02B")
    dwg.add(dwg.text("".join(doras), (337, 400), style="font-size:40;font-family:GL-MahjongTile;"))

    # bou
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "1000_mini.svg"), "rb"
    ) as im:
        b64_1000_mini = base64.b64encode(im.read())

    thousand_mini_img = dwg.image("data:image/svg;base64," + b64_1000_mini.decode("ascii"))
    thousand_mini_img.translate(335, 405)
    thousand_mini_img.scale(0.15)
    dwg.add(thousand_mini_img)
    dwg.add(
        dwg.text(
            f"×{sample_data.riichi}",
            (355, 430),
            style="font-size:22;font-family:serif;",
        )
    )
    hundred_mini_img = dwg.image(
        "https://raw.githubusercontent.com/mjx-project/mjx/master/pymjx/mjx/visualizer/100_mini.svg"
    )
    hundred_mini_img.translate(405, 405)
    hundred_mini_img.scale(0.15)
    dwg.add(hundred_mini_img)
    dwg.add(dwg.text(f"×{sample_data.honba}", (425, 430), style="font-size:22;font-family:serif;"))

    # yama_nokori
    wall_num = sample_data.wall_num
    dwg.add(
        dwg.text(
            "\U0001F02B",
            (370, 465),
            style="font-size:30;font-family:GL-MahjongTile;",
        )
    )
    dwg.add(
        dwg.text(
            "×" + str(wall_num),
            (390, 463),
            style="font-size:22;font-family:serif;",
        )
    )

    for i in range(4):
        # wind
        player_info[i].add(dwg.text(winds[i], (280, 515), style="font-size:30;font-family:serif;"))

        # score
        if len(scores[i]) > 6:
            player_info[i].add(
                dwg.text(
                    scores[i].replace("(", " ("),
                    (335, 490),
                    style="font-size:20;font-family:serif;",
                )
            )
        else:
            player_info[i].add(
                dwg.text(scores[i], (370, 490), style="font-size:20;font-family:serif;")
            )

        # riichi_bou
        if is_riichi[i]:
            thousand_img = dwg.image(
                href="https://raw.githubusercontent.com/mjx-project/mjx/master/pymjx/mjx/visualizer/1000.svg"
            )
            thousand_img.translate(476, 485)
            thousand_img.scale(0.4)
            thousand_img.rotate(90)
            player_info[i].add(thousand_img)

        left_margin = 190
        # hand
        for j, hand in enumerate(hands[i]):
            hand_txt = hand[0]
            dwg_add(dwg, pai[i], (left_margin + j * char_width, 770), hand_txt)

        # discard
        riichi_idx = 100000
        for j, discard in enumerate(discards[i]):
            discard_txt = discard[0]
            if discard[1]:  # riichi
                riichi_idx = j
                dwg_add(
                    dwg,
                    pai[i],
                    (535 + (j // 6) * char_height, -307 - (j % 6) * char_width),
                    discard_txt,
                    rotate=True,
                    transparent=discard[3],  # 鳴かれた
                )

                if discard[2]:  # tsumogiri
                    dwg_add(
                        dwg,
                        pai[i],
                        (535 + (j // 6) * char_height, -307 - (j % 6) * char_width),
                        ["\U0001F02B", False],
                        rotate=True,
                        transparent=discard[3],
                    )

            elif (riichi_idx < j) and (j // 6 == riichi_idx // 6):
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        304 + char_height - char_width + (j % 6) * char_width,
                        570 + (j // 6) * char_height,
                    ),
                    discard_txt,
                    transparent=discard[3],
                )

                if discard[2]:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            304 + char_height - char_width + (j % 6) * char_width,
                            570 + (j // 6) * char_height,
                        ),
                        ["\U0001F02B", False],
                        transparent=discard[3],
                    )
            else:
                dwg_add(
                    dwg,
                    pai[i],
                    (304 + (j % 6) * char_width, 570 + (j // 6) * char_height),
                    discard_txt,
                    transparent=discard[3],
                )

                if discard[2]:
                    dwg_add(
                        dwg,
                        pai[i],
                        (304 + (j % 6) * char_width, 570 + (j // 6) * char_height),
                        ["\U0001F02B", False],
                        transparent=discard[3],
                    )

        num_of_tehai = len(hands[i]) + len(open_tiles[i]) * 3
        left_x = char_width if num_of_tehai == 13 else 0

        for open_tile in open_tiles[i]:
            if open_tile[2] == TileUnitType.CHI:
                chi_txt = open_tile[0]
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - left_x,  # 他の鳴き牌の分のずれ
                    ),
                    chi_txt[0],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                        770,
                    ),
                    chi_txt[1],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + left_x
                        + char_height
                        + char_width,
                        770,
                    ),
                    chi_txt[2],
                )

                left_x += char_width * 2 + char_height

            elif open_tile[2] == TileUnitType.PON:
                pon_txt = open_tile[0]
                if open_tile[1] == FromWho.LEFT:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x,
                        ),
                        pon_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                            770,
                        ),
                        pon_txt[1],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width,
                            770,
                        ),
                        pon_txt[2],
                    )

                elif open_tile[1] == FromWho.MID:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width,  # 1つ目の牌の分のずれ
                        ),
                        pon_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        pon_txt[1],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width,
                            770,
                        ),
                        pon_txt[2],
                    )

                elif open_tile[1] == FromWho.RIGHT:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width  # 1つ目の牌の分のずれ
                            - char_width,  # 2つ目の牌の分のずれ
                        ),
                        pon_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        pon_txt[1],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                            770,
                        ),
                        pon_txt[2],
                    )
                left_x += char_width * 2 + char_height

            elif open_tile[2] == TileUnitType.CLOSED_KAN:
                closed_kan_txt = open_tile[0]
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x,
                        770,
                    ),
                    ["\U0001F02B", False],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                        770,
                    ),
                    closed_kan_txt[1],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width * 2,
                        770,
                    ),
                    closed_kan_txt[2],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width * 3,
                        770,
                    ),
                    ["\U0001F02B", False],
                )
                left_x += char_width * 4

            elif open_tile[2] == TileUnitType.OPEN_KAN:
                open_tile_kan_txt = open_tile[0]
                if open_tile[1] == FromWho.LEFT:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x,
                        ),
                        open_tile_kan_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                            770,
                        ),
                        open_tile_kan_txt[1],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width,
                            770,
                        ),
                        open_tile_kan_txt[2],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width * 2,
                            770,
                        ),
                        open_tile_kan_txt[3],
                    )

                elif open_tile[1] == FromWho.MID:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width,
                        ),
                        open_tile_kan_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        open_tile_kan_txt[1],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width,
                            770,
                        ),
                        open_tile_kan_txt[2],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width * 2,
                            770,
                        ),
                        open_tile_kan_txt[3],
                    )

                elif open_tile[1] == FromWho.RIGHT:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width * 3,
                        ),
                        open_tile_kan_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        open_tile_kan_txt[1],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                            770,
                        ),
                        open_tile_kan_txt[2],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width * 2,
                            770,
                        ),
                        open_tile_kan_txt[3],
                    )
                left_x += char_width * 3 + char_height

            elif open_tile[2] == TileUnitType.ADDED_KAN:
                added_kan_txt = open_tile[0]
                if open_tile[1] == FromWho.LEFT:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x,
                        ),
                        added_kan_txt[0],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            710,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x,
                        ),
                        added_kan_txt[1],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                            770,
                        ),
                        added_kan_txt[2],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width,
                            770,
                        ),
                        added_kan_txt[3],
                    )

                elif open_tile[1] == FromWho.MID:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width,
                        ),
                        added_kan_txt[1],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            710,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width,
                        ),
                        added_kan_txt[2],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        added_kan_txt[0],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_height
                            + char_width,
                            770,
                        ),
                        added_kan_txt[3],
                    )

                elif open_tile[1] == FromWho.RIGHT:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            741.3,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width * 2,
                        ),
                        added_kan_txt[2],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            710,
                            -left_margin  # 初期位置
                            - 3  # 回転時のずれ
                            - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                            - left_x
                            - char_width * 2,
                        ),
                        added_kan_txt[3],
                        rotate=True,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        added_kan_txt[0],
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                            770,
                        ),
                        added_kan_txt[1],
                    )
                left_x += char_width * 2 + char_height

        players[i].add(pai[i])
        players[i].add(player_info[i])
        players[i].rotate((360 - i * 90), (width / 2, height / 2))
        player_g.add(players[i])

    dwg.add(player_g)
    dwg.save()
