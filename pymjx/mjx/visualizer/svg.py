import os
import svgwrite
from mjx.visualizer.visualizer import *


def dwg_add(dwg_p, dwg_g, pos, txt, rotate=False):
    if rotate:
        if txt == "\U0001F004\uFE0E":
            horizontal_tiles = [
                dwg_p.text(txt, insert=(0, 0), fill="red"),
                dwg_p.text(
                    "\U0001F006",
                    insert=(0, 0),
                    stroke=svgwrite.rgb(255, 255, 255, "%"),
                    fill="white",
                ),
                dwg_p.text("\U0001F006", insert=(0, 0), fill="black"),
            ]

            for horizontal_tile in horizontal_tiles:
                horizontal_tile.rotate(90, (0, 0))
                horizontal_tile.translate(pos)
                dwg_g.add(horizontal_tile)
        else:
            horizontal_tile = dwg_p.text(txt, insert=(0, 0))
            horizontal_tile.rotate(90, (0, 0))
            horizontal_tile.translate(pos)
            dwg_g.add(horizontal_tile)
    else:
        if txt == "\U0001F004\uFE0E":
            dwg_g.add(dwg_p.text(txt, pos, fill="red"))
            dwg_g.add(
                dwg_p.text(
                    "\U0001F006", pos, stroke=svgwrite.rgb(255, 255, 255, "%"), fill="white"
                )
            )
            dwg_g.add(dwg_p.text("\U0001F006", pos, fill="black"))
        else:
            dwg_g.add(dwg_p.text(txt, pos))


def make_svg(filename: str, mode: str, page: int):
    width = 800
    height = 800
    char_width = 32  # 45:28.8,60:38.4
    char_height = 44  # 45:40.5,60:53
    chi_width = char_width * 2 + char_height

    data = MahjongTable.load_data(filename, mode)
    sample_data = data[page]
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
    chis = [[], [], [], []]
    pons = [[], [], [], []]
    closed_kans = [[], [], [], []]
    open_kans = [[], [], [], []]
    added_kans = [[], [], [], []]
    discards = [[], [], [], []]

    for i in range(4):  # iは各プレイヤー(0-3)
        players[i] = dwg.g()
        pai[i] = dwg.g(style="font-size:50;font-family:GL-MahjongTile;")
        player_info[i] = dwg.g()

        winds[i] = get_wind_char(sample_data.players[i].wind, lang=1)
        scores[i] = sample_data.players[i].score
        is_riichi[i] = sample_data.players[i].is_declared_riichi

        for t_u in sample_data.players[i].tile_units:
            if t_u.tile_unit_type == TileUnitType.HAND:
                hands[i] = [
                    "\U0001F02B" if not tile.is_open else get_tile_char(tile.id, True)
                    for tile in t_u.tiles
                ]

            if t_u.tile_unit_type == TileUnitType.DISCARD:
                for tile in t_u.tiles:
                    discards[i].append(
                        [
                            get_tile_char(tile.id, True),
                            tile.with_riichi,
                            tile.is_tsumogiri,
                        ]
                    )
            if t_u.tile_unit_type == TileUnitType.CHI:
                chis[i].append(
                    [
                        [get_tile_char(tile.id, True) for tile in t_u.tiles],
                        t_u.from_who,
                    ]
                )
            if t_u.tile_unit_type == TileUnitType.PON:
                pons[i].append(
                    [
                        [get_tile_char(tile.id, True) for tile in t_u.tiles],
                        t_u.from_who,
                    ]
                )
            if t_u.tile_unit_type == TileUnitType.CLOSED_KAN:
                closed_kans[i].append(
                    [
                        [get_tile_char(tile.id, True) for tile in t_u.tiles],
                        t_u.from_who,
                    ]
                )
            if t_u.tile_unit_type == TileUnitType.OPEN_KAN:
                open_kans[i].append(
                    [
                        [get_tile_char(tile.id, True) for tile in t_u.tiles],
                        t_u.from_who,
                    ]
                )
            if t_u.tile_unit_type == TileUnitType.ADDED_KAN:
                added_kans[i].append(
                    [
                        [get_tile_char(tile.id, True) for tile in t_u.tiles],
                        t_u.from_who,
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
    thousand_mini_img = dwg.image("http://drive.google.com/uc?id=12TtHohmEvylFUqSmvCQzG1hO7hdErG6S")
    thousand_mini_img.translate(335, 405)
    thousand_mini_img.scale(0.15)
    dwg.add(thousand_mini_img)
    dwg.add(dwg.text(f"×{sample_data.riichi}",(355, 430),style="font-size:22;font-family:serif;",))
    hundred_mini_img = dwg.image("http://drive.google.com/uc?id=13v91ayZQXzXMM0uMKRPoa9MqIq-x7IHy")
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
                href="http://drive.google.com/uc?id=1UZk4EgZudG7Xv7SxkPPRrxRf9vDW657P"
            )
            thousand_img.translate(476, 485)
            thousand_img.scale(0.4)
            thousand_img.rotate(90)
            player_info[i].add(thousand_img)

        left_margin = (
            width
            - (
                char_width * len(hands[i])
                + (char_width * 3 + char_height)
                * (len(chis[i]) + len(pons[i]) + len(added_kans[i]))
                + (char_width * 5) * len(closed_kans[i])
                + (char_width * 4 + char_height) * len(open_kans[i])
            )
        ) / 2
        # hand
        for j, hand in enumerate(hands[i]):
            dwg_add(dwg, pai[i], (left_margin + j * char_width, 770), hand)

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
                )

                if discard[2]: # tsumogiri
                    dwg_add(
                        dwg,
                        pai[i],
                        (535 + (j // 6) * char_height, -307 - (j % 6) * char_width),
                        "\U0001F02B",
                        rotate=True,
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
                )

                if discard[2]:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            304 + char_height - char_width + (j % 6) * char_width,
                            570 + (j // 6) * char_height,
                        ),
                        "\U0001F02B",
                    )
            else:
                dwg_add(
                    dwg,
                    pai[i],
                    (304 + (j % 6) * char_width, 570 + (j // 6) * char_height),
                    discard_txt,
                )

                if discard[2]:
                    dwg_add(
                        dwg,
                        pai[i],
                        (304 + (j % 6) * char_width, 570 + (j // 6) * char_height),
                        "\U0001F02B",
                    )

        # chi
        for j, chi in enumerate(chis[i]):
            if chis[i] == []:
                continue
            chi_txt = chi[0]
            dwg_add(
                dwg,
                pai[i],
                (
                    741.3,
                    -left_margin  # 初期位置
                    - 3  # 回転時のずれ
                    - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                    - j * (chi_width + char_width),  # 他のチーの分のずれ
                ),
                chi_txt[0],
                True,
            )
            dwg_add(
                dwg,
                pai[i],
                (
                    left_margin
                    + (len(hands[i]) + 1) * char_width
                    + j * (chi_width + char_width)
                    + char_height,
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
                    + j * (chi_width + char_width)
                    + char_height
                    + char_width,
                    770,
                ),
                chi_txt[2],
            )

        # pon
        for j, pon in enumerate(pons[i]):
            if pons[i] == []:
                continue

            pon_txt = pon[0]
            if pon[1] == FromWho.LEFT:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - j * (chi_width + char_width),  # 他のポンの分のずれ
                    ),
                    pon_txt[0],
                    True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + j * (chi_width + char_width)
                        + char_height,
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
                        + (len(chis[i])) * (chi_width + char_width)
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width,
                        770,
                    ),
                    pon_txt[2],
                )

            elif pon[1] == FromWho.MID:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - j * (chi_width + char_width)  # 他のポンの分のずれ
                        - char_width,  # 1つ目の牌の分のずれ
                    ),
                    pon_txt[1],
                    True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + j * (chi_width + char_width),
                        770,
                    ),
                    pon_txt[0],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width,
                        770,
                    ),
                    pon_txt[2],
                )

            elif pon[1] == FromWho.RIGHT:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - j * (chi_width + char_width)  # 他のポンの分のずれ
                        - char_width  # 1つ目の牌の分のずれ
                        - char_width,  # 2つ目の牌の分のずれ
                    ),
                    pon_txt[2],
                    True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + j * (chi_width + char_width),
                        770,
                    ),
                    pon_txt[0],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + j * (chi_width + char_width)
                        + char_width,
                        770,
                    ),
                    pon_txt[1],
                )

        # closed_kan
        for j, closed_kan in enumerate(closed_kans[i]):
            if closed_kans[i] == []:
                continue

            closed_kan_txt = closed_kan[0]

            dwg_add(
                dwg,
                pai[i],
                (
                    left_margin
                    + (len(hands[i]) + 1) * char_width
                    + (len(chis[i])) * (chi_width + char_width)
                    + (len(pons[i])) * (chi_width + char_width)
                    + j * (chi_width + char_width),
                    770,
                ),
                "\U0001F02B",
            )
            dwg_add(
                dwg,
                pai[i],
                (
                    left_margin
                    + (len(hands[i]) + 1) * char_width
                    + (len(chis[i])) * (chi_width + char_width)
                    + (len(pons[i])) * (chi_width + char_width)
                    + j * (chi_width + char_width)
                    + char_width,
                    770,
                ),
                closed_kan_txt[1],
            )
            dwg_add(
                dwg,
                pai[i],
                (
                    left_margin
                    + (len(hands[i]) + 1) * char_width
                    + (len(chis[i])) * (chi_width + char_width)
                    + (len(pons[i])) * (chi_width + char_width)
                    + j * (chi_width + char_width)
                    + char_width * 2,
                    770,
                ),
                closed_kan_txt[2],
            )
            dwg_add(
                dwg,
                pai[i],
                (
                    left_margin
                    + (len(hands[i]) + 1) * char_width
                    + (len(chis[i])) * (chi_width + char_width)
                    + (len(pons[i])) * (chi_width + char_width)
                    + j * (chi_width + char_width)
                    + char_width * 3,
                    770,
                ),
                "\U0001F02B",
            )

        # open_kan
        for j, open_kan in enumerate(open_kans[i]):
            if open_kans[i] == []:
                continue

            open_kan_txt = open_kan[0]
            if open_kan[1] == FromWho.LEFT:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - j * (chi_width + char_width),
                    ),
                    open_kan_txt[0],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_height,
                        770,
                    ),
                    open_kan_txt[1],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width,
                        770,
                    ),
                    open_kan_txt[2],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width * 2,
                        770,
                    ),
                    open_kan_txt[3],
                )

            elif open_kan[1] == FromWho.MID:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - j * (chi_width + char_width)
                        - char_width,
                    ),
                    open_kan_txt[1],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width),
                        770,
                    ),
                    open_kan_txt[0],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width,
                        770,
                    ),
                    open_kan_txt[2],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width * 2,
                        770,
                    ),
                    open_kan_txt[3],
                )

            elif open_kan[1] == FromWho.RIGHT:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - j * (chi_width + char_width)
                        - char_width * 3,
                    ),
                    open_kan_txt[3],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width),
                        770,
                    ),
                    open_kan_txt[0],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_width,
                        770,
                    ),
                    open_kan_txt[1],
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width * 2,
                        770,
                    ),
                    open_kan_txt[2],
                )

        # added_kan
        for j, added_kan in enumerate(added_kans[i]):
            if added_kans[i] == []:
                continue

            added_kan_txt = added_kan[0]
            if added_kan[1] == FromWho.LEFT:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - (len(open_kans[i])) * (char_height + char_width * 4)  # open_kansのずれ
                        - j * (chi_width + char_width),
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
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - (len(open_kans[i])) * (char_height + char_width * 4)  # open_kansのずれ
                        - j * (chi_width + char_width),
                    ),
                    added_kan_txt[1],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + (len(open_kans[i])) * (char_height + char_width * 4)
                        + j * (chi_width + char_width)
                        + char_height,
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
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + (len(open_kans[i])) * (char_height + char_width * 4)
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width,
                        770,
                    ),
                    added_kan_txt[3],
                )

            elif added_kan[1] == FromWho.MID:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - (len(open_kans[i])) * (char_height + char_width * 4)  # open_kansのずれ
                        - j * (chi_width + char_width)
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
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - (len(open_kans[i])) * (char_height + char_width * 4)  # open_kansのずれ
                        - j * (chi_width + char_width)
                        - char_width,
                    ),
                    added_kan_txt[2],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + (len(open_kans[i])) * (char_height + char_width * 4)
                        + j * (chi_width + char_width),
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
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + (len(open_kans[i])) * (char_height + char_width * 4)
                        + j * (chi_width + char_width)
                        + char_height
                        + char_width,
                        770,
                    ),
                    added_kan_txt[3],
                )

            elif added_kan[1] == FromWho.RIGHT:
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        741.3,
                        -left_margin  # 初期位置
                        - 3  # 回転時のずれ
                        - (len(hands[i]) + 1) * char_width  # 手牌の分のずれ
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - (len(open_kans[i])) * (char_height + char_width * 4)  # open_kansのずれ
                        - j * (chi_width + char_width)
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
                        - (len(chis[i])) * (chi_width + char_width)  # チーの分のずれ
                        - (len(pons[i])) * (chi_width + char_width)  # ポンの分のずれ
                        - (len(closed_kans[i])) * char_width * 5  # closed_kansのずれ
                        - (len(open_kans[i])) * (char_height + char_width * 4)  # open_kansのずれ
                        - j * (chi_width + char_width)
                        - char_width * 2,
                    ),
                    added_kan_txt[3],
                    rotate=True,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin
                        + (len(hands[i]) + 1) * char_width
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + (len(open_kans[i])) * (char_height + char_width * 4)
                        + j * (chi_width + char_width),
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
                        + (len(chis[i])) * (chi_width + char_width)
                        + (len(pons[i])) * (chi_width + char_width)
                        + (len(closed_kans[i])) * char_width * 5
                        + (len(open_kans[i])) * (char_height + char_width * 4)
                        + j * (chi_width + char_width)
                        + char_width,
                        770,
                    ),
                    added_kan_txt[1],
                )

        players[i].add(pai[i])
        players[i].add(player_info[i])
        players[i].rotate((360 - i * 90), (width / 2, height / 2))
        player_g.add(players[i])

    dwg.add(player_g)
    dwg.save()
