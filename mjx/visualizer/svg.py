import base64
from importlib.resources import read_binary
from typing import List, Optional, Tuple, Union

import svgwrite
from svgwrite.drawing import Drawing

import mjx.visualizer
import mjxproto
from mjx.const import EventType as TileUnitType
from mjx.const import RelativePlayerIdx
from mjx.visualizer.visualizer import MahjongTable, Tile, get_tile_char, get_wind_char


def dwg_add(
    dwg_p,
    dwg_g,
    pos,
    txt: str,
    rotate: bool = False,
    is_red: bool = False,
    transparent: bool = False,
    highliting: bool = False,
):
    opacity = 1.0
    if transparent:
        opacity = 0.5

    highlight_fill = "black"
    highlight_opacity = 1.0
    highlight_stroke_width = 2.0

    if rotate:
        if is_red:
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

        if highliting:
            highlighted_tile = dwg_p.text(
                "\U0001F006",
                insert=(0, 0),
                stroke=svgwrite.rgb(0, 0, 0, "%"),
                stroke_width=highlight_stroke_width,
                fill=highlight_fill,
                opacity=highlight_opacity,
            )
            highlighted_tile.rotate(90, (0, 0))
            highlighted_tile.translate(pos)
            dwg_g.add(highlighted_tile)

    else:
        if is_red:
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

        if highliting:
            dwg_g.add(
                dwg_p.text(
                    "\U0001F006",
                    pos,
                    stroke=svgwrite.rgb(0, 0, 0, "%"),  # 255,140,0
                    stroke_width=highlight_stroke_width,
                    fill=highlight_fill,
                    opacity=highlight_opacity,
                )
            )


def _make_svg(
    proto_data: Union[mjxproto.State, mjxproto.Observation],
    target_idx: Optional[int] = None,
    show_name: bool = False,
    highlight_last_event: bool = False,
) -> Drawing:
    sample_data: MahjongTable
    if isinstance(proto_data, mjxproto.Observation):
        sample_data = MahjongTable.decode_observation(proto_data)
    else:
        sample_data = MahjongTable.decode_state(proto_data)

    width = 800
    height = 800
    char_width = 32  # 45:28.8,60:38.4
    char_height = 44  # 45:40.5,60:53
    red_hai = [16, 52, 88]

    if target_idx is None:
        if isinstance(proto_data, mjxproto.Observation):
            target_idx = proto_data.who
        else:
            target_idx = 0
    assert target_idx is not None
    sample_data.players.sort(key=lambda x: (x.player_idx - target_idx) % 4)

    dwg = svgwrite.Drawing(
        "temp.svg",  # ファイル名は"saveas()"で指定する
        (width, height),
        debug=True,
    )

    dwg._embed_font_data(
        "GL-MahjongTile",
        read_binary(mjx.visualizer, "GL-MahjongTile.ttf"),
        "application/x-font-ttf",
    )

    player_g = dwg.g()

    players: List[Drawing] = [dwg.g(), dwg.g(), dwg.g(), dwg.g()]
    pai: List[Drawing] = [dwg.g(), dwg.g(), dwg.g(), dwg.g()]
    player_info: List[Drawing] = [dwg.g(), dwg.g(), dwg.g(), dwg.g()]
    winds: List[str] = ["", "", "", ""]
    scores: List[str] = ["", "", "", ""]
    is_riichi = [False, False, False, False]

    # Tuple[char, is_red, Tile]
    hands: List[List[Tuple[str, bool, Tile]]] = [[], [], [], []]
    open_tiles: List[
        List[Tuple[List[Tuple[str, bool, Tile]], Optional[RelativePlayerIdx], TileUnitType]]
    ] = [
        [],
        [],
        [],
        [],
    ]
    discards: List[List[Tuple[str, bool, Tile]]] = [[], [], [], []]

    for i in range(4):  # iは各プレイヤー(0-3)
        players[i] = dwg.g()
        pai[i] = dwg.g(style="font-size:50px;font-family:GL-MahjongTile;")
        player_info[i] = dwg.g()

        winds[i] = get_wind_char(sample_data.players[i].wind, lang=1)
        scores[i] = sample_data.players[i].score
        is_riichi[i] = sample_data.players[i].is_declared_riichi

        for t_u in reversed(sample_data.players[i].tile_units):
            if t_u.tile_unit_type == TileUnitType.DRAW:
                for tile in t_u.tiles:
                    hands[i].append(
                        (
                            "\U0001F02B" if not tile.is_open else get_tile_char(tile.id(), True),
                            tile.id in red_hai,
                            tile,
                        )
                    )

            if t_u.tile_unit_type == TileUnitType.DISCARD:
                for tile in t_u.tiles:
                    discards[i].append((get_tile_char(tile.id(), True), tile.id in red_hai, tile))
            if t_u.tile_unit_type in [
                TileUnitType.CHI,
                TileUnitType.PON,
                TileUnitType.CLOSED_KAN,
                TileUnitType.OPEN_KAN,
                TileUnitType.ADDED_KAN,
            ]:
                open_tiles[i].append(
                    (
                        [
                            (get_tile_char(tile.id(), True), tile.id in red_hai, tile)
                            for tile in t_u.tiles
                        ],
                        t_u.from_who,
                        t_u.tile_unit_type,
                    )
                )
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height)))
    dwg.add(dwg.rect(insert=(1, 1), size=(width - 2, height - 2), fill="rgb(255,255,255)"))

    dwg.add(dwg.rect(insert=(278, 278), size=(244, 244)))
    dwg.add(dwg.rect(insert=(279, 279), size=(242, 242), fill="rgb(255,255,255)"))

    # board_info
    if sample_data.honba > 0:
        round = (
            get_wind_char((sample_data.round - 1) // 4, lang=1)
            + str((sample_data.round - 1) % 4 + 1)
            + "局"
        )
        dwg.add(dwg.text(round, (340, 360), style="font-size:24px;font-family:serif;"))
        honba = str(sample_data.honba) + "本場"
        dwg.add(dwg.text(honba, (400, 360), style="font-size:24px;font-family:serif;"))
    else:
        round = (
            get_wind_char((sample_data.round - 1) // 4, lang=1)
            + str((sample_data.round - 1) % 4 + 1)
            + "局"
        )
        dwg.add(dwg.text(round, (368, 360), style="font-size:26px;font-family:serif;"))

    # dora
    doras = [get_tile_char(tile, True) for tile in sample_data.doras]
    while len(doras) < 5:
        doras.append("\U0001F02B")
    dwg.add(
        dwg.text("".join(doras), (337, 400), style="font-size:40px;font-family:GL-MahjongTile;")
    )
    for i, dora in enumerate(sample_data.doras):
        if dora == sample_data.new_dora and highlight_last_event:
            dwg.add(
                dwg.text(
                    "\U0001F006",
                    (337 + i * 25.6, 400),
                    style="font-size:40px;font-family:GL-MahjongTile;",
                    stroke=svgwrite.rgb(0, 0, 0, "%"),
                    stroke_width=2.0,
                )
            )

    # bou
    b64_1000_mini = base64.b64encode(
        read_binary(mjx.visualizer, "1000_mini.svg"),
    )
    thousand_mini_img = dwg.image("data:image/svg+xml;base64," + b64_1000_mini.decode("ascii"))
    thousand_mini_img.translate(335, 405)
    thousand_mini_img.scale(0.15)
    dwg.add(thousand_mini_img)
    dwg.add(
        dwg.text(
            f"×{sample_data.riichi}",
            (355, 430),
            style="font-size:22px;font-family:serif;",
        )
    )

    b64_hundred_mini = base64.b64encode(
        read_binary(mjx.visualizer, "100_mini.svg"),
    )
    hundred_mini_img = dwg.image("data:image/svg+xml;base64," + b64_hundred_mini.decode("ascii"))
    hundred_mini_img.translate(405, 405)
    hundred_mini_img.scale(0.15)
    dwg.add(hundred_mini_img)
    dwg.add(
        dwg.text(f"×{sample_data.honba}", (425, 430), style="font-size:22px;font-family:serif;")
    )

    # yama_nokori
    wall_num = sample_data.wall_num
    dwg.add(
        dwg.text(
            "\U0001F02B",
            (370, 465),
            style="font-size:30px;font-family:GL-MahjongTile;",
        )
    )
    dwg.add(
        dwg.text(
            "×" + str(wall_num),
            (390, 463),
            style="font-size:22px;font-family:serif;",
        )
    )

    for i in range(4):
        # name
        if show_name:
            player_info[i].add(
                dwg.text(
                    sample_data.players[i].name,
                    (10, 790),
                    style="font-size:18px;font-family:serif;",
                )
            )

        # wind
        player_info[i].add(
            dwg.text(winds[i], (280, 515), style="font-size:30px;font-family:serif;")
        )

        # score
        scores[i] = scores[i].replace("(", " (")
        score_x = width / 2 - len(scores[i]) / 2 * 11
        player_info[i].add(
            dwg.text(scores[i], (score_x, 490), style="font-size:20px;font-family:serif;")
        )

        # riichi_bou
        if is_riichi[i]:
            b64_thousand = base64.b64encode(
                read_binary(mjx.visualizer, "1000.svg"),
            )
            thousand_img = dwg.image("data:image/svg+xml;base64," + b64_thousand.decode("ascii"))
            thousand_img.translate(476, 485)
            thousand_img.scale(0.4)
            thousand_img.rotate(90)
            player_info[i].add(thousand_img)

        left_margin = 190
        # hand
        for j, hand in enumerate(hands[i]):
            hand_txt = hand[0]
            dwg_add(
                dwg,
                pai[i],
                (left_margin + j * char_width, 770),
                hand_txt,
                highliting=hand[2].is_highlighting and highlight_last_event,
            )

        # discard
        riichi_idx = 100000
        for j, discard in enumerate(discards[i]):
            discard_txt = discard[0]
            if discard[2].with_riichi:  # riichi
                riichi_idx = j
                dwg_add(
                    dwg,
                    pai[i],
                    (535 + (j // 6) * char_height, -307 - (j % 6) * char_width),
                    discard_txt,
                    rotate=True,
                    is_red=discard[1],
                    transparent=discard[2].is_transparent,  # 鳴かれた
                    highliting=discard[2].is_highlighting and highlight_last_event,
                )

                if discard[2].is_tsumogiri:
                    dwg_add(
                        dwg,
                        pai[i],
                        (535 + (j // 6) * char_height, -307 - (j % 6) * char_width),
                        "\U0001F02B",
                        rotate=True,
                        transparent=discard[2].is_transparent,
                        highliting=discard[2].is_highlighting and highlight_last_event,
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
                    is_red=discard[1],
                    transparent=discard[2].is_transparent,
                    highliting=discard[2].is_highlighting and highlight_last_event,
                )

                if discard[2].is_tsumogiri:
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            304 + char_height - char_width + (j % 6) * char_width,
                            570 + (j // 6) * char_height,
                        ),
                        "\U0001F02B",
                        transparent=discard[2].is_transparent,
                        highliting=discard[2].is_highlighting and highlight_last_event,
                    )
            else:
                dwg_add(
                    dwg,
                    pai[i],
                    (304 + (j % 6) * char_width, 570 + (j // 6) * char_height),
                    discard_txt,
                    is_red=discard[1],
                    transparent=discard[2].is_transparent,
                    highliting=discard[2].is_highlighting and highlight_last_event,
                )

                if discard[2].is_tsumogiri:
                    dwg_add(
                        dwg,
                        pai[i],
                        (304 + (j % 6) * char_width, 570 + (j // 6) * char_height),
                        "\U0001F02B",
                        transparent=discard[2].is_transparent,
                        highliting=discard[2].is_highlighting and highlight_last_event,
                    )

        num_of_tehai = len(hands[i]) + len(open_tiles[i]) * 3
        left_x = char_width if num_of_tehai == 13 else 0

        for open_tile in open_tiles[i]:
            if open_tile[2] == TileUnitType.CHI:
                chi = open_tile[0]
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
                    chi[0][0],
                    rotate=True,
                    is_red=chi[0][1],
                    highliting=chi[0][2].is_highlighting and highlight_last_event,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                        770,
                    ),
                    chi[1][0],
                    is_red=chi[1][1],
                    highliting=chi[0][2].is_highlighting and highlight_last_event,
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
                    chi[2][0],
                    is_red=chi[2][1],
                    highliting=chi[0][2].is_highlighting and highlight_last_event,
                )

                left_x += char_width * 2 + char_height

            elif open_tile[2] == TileUnitType.PON:
                pon = open_tile[0]
                if open_tile[1] == RelativePlayerIdx.LEFT:
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
                        pon[0][0],
                        rotate=True,
                        is_red=pon[0][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                            770,
                        ),
                        pon[1][0],
                        is_red=pon[1][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
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
                        pon[2][0],
                        is_red=pon[2][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )

                elif open_tile[1] == RelativePlayerIdx.CENTER:
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
                        pon[0][0],
                        rotate=True,
                        is_red=pon[0][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        pon[1][0],
                        is_red=pon[1][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
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
                        pon[2][0],
                        is_red=pon[2][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )

                elif open_tile[1] == RelativePlayerIdx.RIGHT:
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
                        pon[0][0],
                        rotate=True,
                        is_red=pon[0][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        pon[1][0],
                        is_red=pon[1][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                            770,
                        ),
                        pon[2][0],
                        is_red=pon[2][1],
                        highliting=pon[0][2].is_highlighting and highlight_last_event,
                    )
                left_x += char_width * 2 + char_height

            elif open_tile[2] == TileUnitType.CLOSED_KAN:
                closed_kan = open_tile[0]
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x,
                        770,
                    ),
                    "\U0001F02B",
                    highliting=closed_kan[0][2].is_highlighting and highlight_last_event,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                        770,
                    ),
                    closed_kan[1][0],
                    is_red=closed_kan[1][1],
                    highliting=closed_kan[0][2].is_highlighting and highlight_last_event,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width * 2,
                        770,
                    ),
                    closed_kan[2][0],
                    is_red=closed_kan[2][1],
                    highliting=closed_kan[0][2].is_highlighting and highlight_last_event,
                )
                dwg_add(
                    dwg,
                    pai[i],
                    (
                        left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width * 3,
                        770,
                    ),
                    "\U0001F02B",
                    highliting=closed_kan[0][2].is_highlighting and highlight_last_event,
                )
                left_x += char_width * 4

            elif open_tile[2] == TileUnitType.OPEN_KAN:
                open_tile_kan = open_tile[0]
                if open_tile[1] == RelativePlayerIdx.LEFT:
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
                        open_tile_kan[0][0],
                        rotate=True,
                        is_red=open_tile_kan[0][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                            770,
                        ),
                        open_tile_kan[1][0],
                        is_red=open_tile_kan[1][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
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
                        open_tile_kan[2][0],
                        is_red=open_tile_kan[2][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
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
                        open_tile_kan[3][0],
                        is_red=open_tile_kan[3][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )

                elif open_tile[1] == RelativePlayerIdx.CENTER:
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
                        txt=open_tile_kan[0][0],
                        rotate=True,
                        is_red=open_tile_kan[0][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        open_tile_kan[1][0],
                        is_red=open_tile_kan[1][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
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
                        open_tile_kan[2][0],
                        is_red=open_tile_kan[2][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
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
                        open_tile_kan[3][0],
                        is_red=open_tile_kan[3][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )

                elif open_tile[1] == RelativePlayerIdx.RIGHT:
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
                        txt=open_tile_kan[0][0],
                        rotate=True,
                        is_red=open_tile_kan[0][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        open_tile_kan[1][0],
                        is_red=open_tile_kan[1][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                            770,
                        ),
                        open_tile_kan[2][0],
                        is_red=open_tile_kan[2][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin
                            + (len(hands[i]) + 1) * char_width
                            + left_x
                            + char_width * 2,
                            770,
                        ),
                        open_tile_kan[3][0],
                        is_red=open_tile_kan[3][1],
                        highliting=open_tile_kan[0][2].is_highlighting and highlight_last_event,
                    )
                left_x += char_width * 3 + char_height

            elif open_tile[2] == TileUnitType.ADDED_KAN:
                added_kan = open_tile[0]
                if open_tile[1] == RelativePlayerIdx.LEFT:
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
                        txt=added_kan[0][0],
                        rotate=True,
                        is_red=added_kan[0][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
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
                        added_kan[1][0],
                        rotate=True,
                        is_red=added_kan[1][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_height,
                            770,
                        ),
                        added_kan[2][0],
                        is_red=added_kan[2][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
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
                        added_kan[3][0],
                        is_red=added_kan[3][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )

                elif open_tile[1] == RelativePlayerIdx.CENTER:
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
                        added_kan[1][0],
                        rotate=True,
                        is_red=added_kan[1][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
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
                        added_kan[2][0],
                        rotate=True,
                        is_red=added_kan[2][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        added_kan[0][0],
                        is_red=added_kan[0][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
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
                        added_kan[3][0],
                        is_red=added_kan[3][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )

                elif open_tile[1] == RelativePlayerIdx.RIGHT:
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
                        added_kan[2][0],
                        rotate=True,
                        is_red=added_kan[2][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
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
                        added_kan[3][0],
                        rotate=True,
                        is_red=added_kan[3][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x,
                            770,
                        ),
                        added_kan[0][0],
                        is_red=added_kan[0][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )
                    dwg_add(
                        dwg,
                        pai[i],
                        (
                            left_margin + (len(hands[i]) + 1) * char_width + left_x + char_width,
                            770,
                        ),
                        added_kan[1][0],
                        is_red=added_kan[1][1],
                        highliting=added_kan[0][2].is_highlighting and highlight_last_event,
                    )
                left_x += char_width * 2 + char_height

        players[i].add(pai[i])
        players[i].add(player_info[i])
        players[i].rotate((360 - i * 90), (width / 2, height / 2))
        player_g.add(players[i])

    dwg.add(player_g)
    return dwg


def to_svg(
    proto_data: Union[mjxproto.State, mjxproto.Observation],
    target_idx: Optional[int] = None,
    highlight_last_event: bool = True,
) -> None:
    dwg = _make_svg(proto_data, target_idx, highlight_last_event=highlight_last_event)
    return dwg.tostring()


def save_svg(
    proto_data: Union[mjxproto.State, mjxproto.Observation],
    filename: str = "temp.svg",
    target_idx: Optional[int] = None,
    highlight_last_event: bool = True,
) -> None:
    """Visualize State/Observation proto and save as svg file.

    Args
    ----
      proto_data: State or observation proto
      target_idx: the player you want to highlight
    """
    dwg = _make_svg(proto_data, target_idx, highlight_last_event=highlight_last_event)
    dwg.saveas(filename=filename)


def show_svg(
    proto_data: Union[mjxproto.State, mjxproto.Observation],
    target_idx: Optional[int] = None,
    highlight_last_event: bool = True,
) -> None:
    import sys

    dwg = _make_svg(proto_data, target_idx, highlight_last_event=highlight_last_event)

    if "ipykernel" in sys.modules:
        # Jupyter Notebook
        from IPython.display import display_svg

        display_svg(dwg.tostring(), raw=True)
    else:
        sys.stdout.write("This function only works in Jupyter Notebook.")
