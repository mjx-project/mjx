import argparse
import os
import mjxproto
from mjxproto import EventType
import open_utils
from converter import TileUnitType, FromWho
from converter import get_modifier, get_tile_char, get_wind_char, get_yaku, get_end_type
from rich import print
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.text import Text


class Tile:
    def __init__(
        self,
        tile_id: int,
        is_open: bool = False,
        is_using_unicode: bool = False,
        is_tsumogiri: bool = False,
        with_riichi: bool = False,
    ):
        self.id = tile_id
        self.is_tsumogiri = is_tsumogiri
        self.with_riichi = with_riichi
        if not is_open:
            self.char = "\U0001F02B" if is_using_unicode else "#"
        else:
            self.char = get_tile_char(tile_id, is_using_unicode)
        if is_tsumogiri:
            if with_riichi:
                if is_using_unicode and self.char != "\U0001F004\uFE0E":
                    self.char += " *r"
                else:
                    self.char += "*r"
            else:
                if is_using_unicode and self.char != "\U0001F004\uFE0E":
                    self.char += " *"
                else:
                    self.char += "*"
        elif with_riichi:
            if is_using_unicode and self.char != "\U0001F004\uFE0E":
                self.char += " r"
            else:
                self.char += "r"


class TileUnit:
    def __init__(
        self,
        tiles_type: TileUnitType,
        from_who: FromWho,
        tiles: list,
    ):
        self.tile_unit_type = tiles_type
        self.from_who = from_who
        self.tiles = tiles


class Player:
    def __init__(self):
        self.player_idx: int
        self.wind: int
        self.score: str
        self.tile_units = [
            TileUnit(
                TileUnitType.DISCARD,
                FromWho.NONE,
                [],
            )
        ]
        self.riichi_now = False
        self.is_declared_riichi = False
        self.name: str


class MahjongTable:
    """
    MahjongTableクラスは場の情報（プレイヤーの手牌や河など）を保持します。
    """

    def __init__(self):
        self.players = [Player() for _ in range(4)]
        self.riichi = 0
        self.round = 0
        self.honba = 0
        self.my_idx = 0  # 0-3; The player you want to show.
        self.wall_num = 134
        self.doras = []
        self.uradoras = []
        self.result = ""
        self.end_info = "\n"

    def check_num_tiles(self) -> bool:
        for p in self.players:
            num_of_tiles = 0
            for tile_unit in p.tile_units:
                if tile_unit.tile_unit_type == TileUnitType.HAND:
                    num_of_tiles += len(tile_unit.tiles)
                elif tile_unit.tile_unit_type != TileUnitType.DISCARD:
                    num_of_tiles += 3

            if num_of_tiles < 13 or 14 < num_of_tiles:
                print("ERROR: The number of tiles is inaccurate.")
                print("player", p.player_idx, ":", num_of_tiles)
                return False
        return True


class GameBoard:
    """
    GameBoardクラスは内部にMahjongTableクラスのオブジェクトを持ち、
    EventHistoryからの現在の状態の読み取りや、その表示などを行います。
    """

    def __init__(
        self,
        path: str,
        mode: str,
        is_using_unicode: bool,
        is_using_rich: bool,
        language: int,
        show_name: bool,
    ):
        self.path = path
        self.mode = mode
        self.is_using_unicode = is_using_unicode
        self.is_using_rich = is_using_rich
        self.language = language
        self.show_name = show_name
        self.my_idx = 0
        self.tables = []

        self.layout = Layout()

        if show_name:
            self.layout.split_column(
                Layout(" ", name="space_top"),
                Layout(name="info"),
                Layout(name="players_info_top"),
                Layout(name="table"),
            )
            self.layout["players_info_top"].size = 7
            self.layout["players_info_top"].split_row(
                Layout(" ", name="player1_info_top"),
                Layout(" ", name="player2_info_top"),
                Layout(" ", name="player3_info_top"),
                Layout(" ", name="player4_info_top"),
            )
        else:
            self.layout.split_column(
                Layout(" ", name="space_top"),
                Layout(name="info"),
                Layout(name="table"),
            )

        self.layout["space_top"].size = 3
        self.layout["info"].size = 3
        self.layout["table"].minimum_size = 20

        self.layout["table"].split_column(
            Layout(name="upper1"),
            Layout(name="middle1"),
            Layout(name="lower1"),
        )

        self.layout["upper1"].size = 3
        self.layout["lower1"].size = 3

        self.layout["upper1"].split_row(
            Layout(" "),
            Layout(" ", name="hand3"),
            Layout(" "),
        )
        self.layout["hand3"].ratio = 6

        self.layout["middle1"].split_row(
            Layout(" ", name="hand4"),
            Layout(name="middle2"),
            Layout(" ", name="hand2"),
        )
        self.layout["middle2"].ratio = 10

        self.layout["lower1"].split_row(
            Layout(" "),
            Layout(" ", name="hand1"),
            Layout(" "),
        )
        self.layout["hand1"].ratio = 6

        self.layout["middle2"].split_column(
            Layout(name="upper2"),
            Layout(name="middle3"),
            Layout(name="lower2"),
        )

        self.layout["upper2"].split_row(
            Layout(" ", name="player3_info_corner"),
            Layout(" ", name="discard3"),
            Layout(" ", name="player2_info_corner"),
        )

        self.layout["middle3"].split_row(
            Layout(" ", name="discard4"),
            Layout(" ", name="middle4"),
            Layout(" ", name="discard2"),
        )

        self.layout["middle4"].split_column(
            Layout(" ", name="space_for_info_center"),
            Layout(" ", name="player3_info_center"),
            Layout(" ", name="middle5"),
            Layout(" ", name="player1_info_center"),
        )
        self.layout["space_for_info_center"].size = 1

        self.layout["middle5"].split_row(
            Layout(" ", name="player4_info_center"),
            Layout(" ", name="player2_info_center"),
        )
        self.layout["lower2"].split_row(
            Layout(" ", name="player4_info_corner"),
            Layout(" ", name="discard1"),
            Layout(" ", name="player1_info_corner"),
        )

    def decode_private_observation(
        self, table: MahjongTable, private_observation, who: int
    ):
        """
        MahjongTableのデータに、
        - 手牌
        の情報を読み込ませる関数
        """
        table.players[who].tile_units.append(
            TileUnit(
                TileUnitType.HAND,
                FromWho.NONE,
                [
                    Tile(i, True, self.is_using_unicode)
                    for i in private_observation.curr_hand.closed_tiles
                ],
            )
        )

        return table

    def decode_public_observation(self, table: MahjongTable, public_observation):
        """
        MahjongTableのデータに、
        - 手牌
        - 鳴き牌
        **以外**の情報を読み込ませる関数
        """
        table.round = public_observation.init_score.round + 1
        table.honba = public_observation.init_score.honba
        table.riichi = public_observation.init_score.riichi
        table.doras = public_observation.dora_indicators

        for i, p in enumerate(table.players):
            p.name = public_observation.player_ids[i]
            p.score = str(public_observation.init_score.tens[i])

        for eve in public_observation.events:
            p = table.players[eve.who]

            if eve.type == EventType.EVENT_TYPE_DISCARD:
                for t_u in p.tile_units:
                    if t_u.tile_unit_type == TileUnitType.DISCARD:
                        if p.riichi_now:
                            t_u.tiles.append(
                                Tile(
                                    eve.tile,
                                    True,
                                    self.is_using_unicode,
                                    with_riichi=True,
                                )
                            )
                            p.riichi_now = False
                        else:
                            t_u.tiles.append(
                                Tile(eve.tile, True, self.is_using_unicode)
                            )

            if eve.type == EventType.EVENT_TYPE_TSUMOGIRI:
                for t_u in p.tile_units:
                    if t_u.tile_unit_type == TileUnitType.DISCARD:
                        if p.riichi_now:
                            t_u.tiles.append(
                                Tile(
                                    eve.tile,
                                    True,
                                    self.is_using_unicode,
                                    True,
                                    True,
                                )
                            )
                            p.riichi_now = False
                        else:
                            t_u.tiles.append(
                                Tile(eve.tile, True, self.is_using_unicode, True)
                            )

            if eve.type == EventType.EVENT_TYPE_RIICHI_SCORE_CHANGE:
                table.riichi += 1
                p.is_declared_riichi = True
                p.riichi_now = True
                p.score = str(int(p.score) - 1000)

            if eve.type == EventType.EVENT_TYPE_RON:
                p = table.players[(eve.who - 1) % 4]
                for t_u in p.tile_units:
                    if t_u.tile_unit_type == TileUnitType.DISCARD:
                        t_u.tiles.pop(-1)

            if eve.type in [
                EventType.EVENT_TYPE_CHI,
                EventType.EVENT_TYPE_PON,
                EventType.EVENT_TYPE_CLOSED_KAN,
                EventType.EVENT_TYPE_ADDED_KAN,
                EventType.EVENT_TYPE_OPEN_KAN,
            ]:
                p.tile_units.append(
                    TileUnit(
                        open_utils.open_event_type(eve.open),
                        open_utils.open_from(eve.open),
                        [
                            Tile(i, True, self.is_using_unicode)
                            for i in open_utils.open_tile_ids(eve.open)
                        ],
                    )
                )

                idx_from = -1
                if open_utils.open_from(eve.open) == FromWho.LEFT:
                    idx_from = (p.player_idx + 3) % 4
                elif open_utils.open_from(eve.open) == FromWho.MID:
                    idx_from = (p.player_idx + 2) % 4
                elif open_utils.open_from(eve.open) == FromWho.RIGHT:
                    idx_from = (p.player_idx + 1) % 4

                p_from = table.players[idx_from]
                for p_from_t_u in p_from.tile_units:
                    if p_from_t_u.tile_unit_type == TileUnitType.DISCARD:
                        p_from_t_u.tiles.pop(-1)

        if public_observation.events == []:
            return table

        if public_observation.events[-1].type in [
            EventType.EVENT_TYPE_TSUMO,
            EventType.EVENT_TYPE_RON,
        ]:
            table.result = "win"
        elif public_observation.events[-1].type in [
            EventType.EVENT_TYPE_ABORTIVE_DRAW_NINE_TERMINALS,
            EventType.EVENT_TYPE_ABORTIVE_DRAW_FOUR_RIICHIS,
            EventType.EVENT_TYPE_ABORTIVE_DRAW_THREE_RONS,
            EventType.EVENT_TYPE_ABORTIVE_DRAW_FOUR_KANS,
            EventType.EVENT_TYPE_ABORTIVE_DRAW_FOUR_WINDS,
            EventType.EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL,
            EventType.EVENT_TYPE_EXHAUSTIVE_DRAW_NAGASHI_MANGAN,
        ]:
            table.result = "nowinner"
            table.end_info = get_end_type(public_observation.events[-1].type)

        return table

    def decode_round_terminal(self, table: MahjongTable, round_terminal, win: bool):
        if win:
            table.uradoras = round_terminal.wins[0].ura_dora_indicators
            winner = table.players[round_terminal.wins[0].who]
            winner.tile_units = [
                i for i in winner.tile_units if i.tile_unit_type == TileUnitType.DISCARD
            ]
            winner.tile_units.append(
                TileUnit(
                    TileUnitType.HAND,
                    FromWho.NONE,
                    [
                        Tile(i, True, self.is_using_unicode)
                        for i in round_terminal.wins[0].hand.closed_tiles
                    ],
                )
            )
            for opens in round_terminal.wins[0].hand.opens:
                winner.tile_units.append(
                    TileUnit(
                        open_utils.open_event_type(opens),
                        open_utils.open_from(opens),
                        [
                            Tile(i, True, self.is_using_unicode)
                            for i in open_utils.open_tile_ids(opens)
                        ],
                    )
                )

            for i, p in enumerate(table.players):
                delta = round_terminal.wins[0].ten_changes[i]
                p.score = (
                    str(int(p.score) + delta)
                    + ("(+" if delta > 0 else "(")
                    + str(delta)
                    + ")"
                )
        else:
            for tenpai in round_terminal.no_winner.tenpais:
                tenpai_p = table.players[tenpai.who]
                tenpai_p.tile_units = [
                    i
                    for i in tenpai_p.tile_units
                    if i.tile_unit_type == TileUnitType.DISCARD
                ]
                tenpai_p.tile_units.append(
                    TileUnit(
                        TileUnitType.HAND,
                        FromWho.NONE,
                        [
                            Tile(i, True, self.is_using_unicode)
                            for i in tenpai.hand.closed_tiles
                        ],
                    )
                )
                for opens in tenpai.hand.opens:
                    tenpai_p.tile_units.append(
                        TileUnit(
                            open_utils.open_event_type(opens),
                            open_utils.open_from(opens),
                            [
                                Tile(i, True, self.is_using_unicode)
                                for i in open_utils.open_tile_ids(opens)
                            ],
                        )
                    )

            for i, p in enumerate(table.players):
                delta = round_terminal.no_winner.ten_changes[i]
                p.score = (
                    str(int(p.score) + delta)
                    + ("(+" if delta > 0 else "(")
                    + str(delta)
                    + ")"
                )

        return table

    def decode_observation(self, jsondata):
        gamedata = mjxproto.Observation()
        gamedata.from_json(jsondata)

        table = MahjongTable()

        table.my_idx = gamedata.who
        for i, p in enumerate(table.players):
            p.player_idx = i
            p.wind = i
        table = self.decode_public_observation(table, gamedata.public_observation)
        table = self.decode_private_observation(
            table, gamedata.private_observation, gamedata.who
        )

        # Obsevationの場合,適切な数の裏向きの手牌を用意する
        for i, p in enumerate(table.players):
            if p.player_idx != table.my_idx:
                hands_num = 13
                for t_u in p.tile_units:
                    if t_u.tile_unit_type not in [
                        TileUnitType.DISCARD,
                        TileUnitType.HAND,
                    ]:
                        hands_num -= 3

                p.tile_units.append(
                    TileUnit(
                        TileUnitType.HAND,
                        FromWho.NONE,
                        [
                            Tile(i, False, self.is_using_unicode)
                            for i in range(hands_num)
                        ],
                    )
                )
            p.wind = (-table.round + 1 + i) % 4

        table.wall_num = self.get_wall_num(table)

        if gamedata.public_observation.events != []:
            if table.result == "win":
                table = self.decode_round_terminal(table, gamedata.round_terminal, True)

            elif table.result == "nowinner":
                table = self.decode_round_terminal(
                    table, gamedata.round_terminal, False
                )

        return table

    def decode_state(self, jsondata):
        gamedata = mjxproto.State()
        gamedata.from_json(jsondata)

        table = MahjongTable()
        for i, p in enumerate(table.players):
            p.player_idx = i
        table.my_idx = 0
        table = self.decode_public_observation(table, gamedata.public_observation)
        for i in range(4):
            table = self.decode_private_observation(
                table, gamedata.private_observations[i], i
            )

        for i, p in enumerate(table.players):
            p.wind = (-table.round + 1 + i) % 4

        table.wall_num = self.get_wall_num(table)

        return table

    def get_wall_num(self, table: MahjongTable) -> int:
        all = 136 - 14
        for p in table.players:
            for t_u in p.tile_units:
                all -= len(t_u.tiles)

        return all

    def load_data(self):
        with open(self.path, "r", errors="ignore") as f:
            self.tables = []
            for line in f:
                if self.mode == "obs":
                    table = self.decode_observation(line)
                    self.tables.append(table)
                elif self.mode == "sta":
                    table = self.decode_state(line)
                    self.tables.append(table)
        return self.tables

    def get_modified_tiles(
        self, table: MahjongTable, player_idx: int, tile_unit_type: TileUnitType
    ):
        if self.is_using_rich:
            tiles = ""
            for tile_unit in table.players[player_idx].tile_units:
                if tile_unit.tile_unit_type == tile_unit_type:
                    if tile_unit.tile_unit_type == TileUnitType.DISCARD:
                        discards = [
                            tile.char
                            + (
                                ""
                                if (
                                    tile.char == "\U0001F004\uFE0E"
                                    or (tile.is_tsumogiri and tile.with_riichi)
                                )
                                else " "
                            )
                            + (
                                ""
                                if (tile.is_tsumogiri or tile.with_riichi)
                                else "  "
                                if self.is_using_unicode
                                else " "
                            )
                            for tile in tile_unit.tiles
                        ]
                        tiles += "\n".join(
                            [
                                "".join(discards[idx : idx + 6])
                                for idx in range(0, len(discards), 6)
                            ]
                        )
                        break
                    if player_idx == 1:
                        tiles += (
                            "\n"
                            + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                            + "\n"
                            + (
                                "\n "
                                if tile_unit.tiles[0].char == "\U0001F004\uFE0E"
                                else "\n"
                            ).join(
                                [
                                    tile.char
                                    for tile in sorted(
                                        tile_unit.tiles,
                                        key=lambda x: x.id,
                                        reverse=True,
                                    )
                                ]
                            )
                            + "\n"
                        )
                    elif player_idx == 2:
                        if tile_unit.tile_unit_type == TileUnitType.HAND:
                            tiles += "".join(
                                [
                                    tile.char
                                    + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                                    for tile in tile_unit.tiles
                                ]
                            )
                            break
                        tiles += get_modifier(
                            tile_unit.from_who, tile_unit.tile_unit_type
                        ) + "".join(
                            [
                                tile.char
                                + (
                                    ""
                                    if (
                                        not self.is_using_unicode
                                        or tile.char == "\U0001F004\uFE0E"
                                    )
                                    else " "
                                )
                                for tile in sorted(
                                    tile_unit.tiles,
                                    key=lambda x: x.id,
                                    reverse=True,
                                )
                            ]
                        )
                    elif player_idx == 3:
                        tiles += (
                            "\n"
                            + "\n".join([tile.char for tile in tile_unit.tiles])
                            + "\n "
                            + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                            + "\n"
                        )

                    else:
                        if tile_unit.tile_unit_type == TileUnitType.HAND:
                            tiles += "".join(
                                [
                                    tile.char
                                    + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                                    for tile in tile_unit.tiles
                                ]
                            )
                            break
                        tiles += (
                            "".join(
                                [
                                    tile.char
                                    + (
                                        ""
                                        if (
                                            not self.is_using_unicode
                                            or tile.char == "\U0001F004\uFE0E"
                                        )
                                        else " "
                                    )
                                    for tile in tile_unit.tiles
                                ]
                            )
                            + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                            + " "
                        )

            return tiles
        else:
            tiles = ""
            for tile_unit in table.players[player_idx].tile_units:
                if tile_unit.tile_unit_type == tile_unit_type:
                    if tile_unit.tile_unit_type == TileUnitType.HAND:
                        tiles += "".join(
                            [
                                tile.char
                                + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                                for tile in tile_unit.tiles
                            ]
                        )
                        break
                    if tile_unit.tile_unit_type == TileUnitType.DISCARD:
                        discards = [
                            tile.char
                            + (
                                ""
                                if (
                                    tile.char == "\U0001F004\uFE0E"
                                    or (tile.is_tsumogiri and tile.with_riichi)
                                )
                                else " "
                            )
                            + (
                                ""
                                if (tile.is_tsumogiri or tile.with_riichi)
                                else "  "
                                if self.is_using_unicode
                                else " "
                            )
                            for tile in tile_unit.tiles
                        ]
                        tiles += "\n".join(
                            [
                                "".join(discards[idx : idx + 6])
                                for idx in range(0, len(discards), 6)
                            ]
                        )
                        break

                    tiles += "".join(
                        [
                            tile.char
                            + (
                                ""
                                if (
                                    not self.is_using_unicode
                                    or tile.char == "\U0001F004\uFE0E"
                                )
                                else " "
                            )
                            for tile in tile_unit.tiles
                        ]
                    ) + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)

            return tiles

    def get_board_info(self, table: MahjongTable) -> str:
        board_info = []
        board_info.append(
            [
                f"round:{table.round}",
                get_wind_char((table.round - 1) // 4, self.language)
                + str((table.round - 1) % 4 + 1)
                + "局",
            ][self.language]
        )
        if table.honba > 0:
            board_info.append(
                " "
                + ["honba:" + str(table.honba), str(table.honba) + "本場"][self.language]
            )
        if table.riichi > 0:
            board_info.append(
                " " + ["riichi:", "供託"][self.language] + str(table.riichi)
            )
        board_info.append(
            " "
            + ["wall:" + str(table.wall_num), "残り:" + str(table.wall_num) + "枚"][
                self.language
            ]
        )
        dora = "".join(
            [
                get_tile_char(d, self.is_using_unicode)
                + (
                    ""
                    if get_tile_char(d, self.is_using_unicode) == "\U0001F004\uFE0E"
                    else " "
                )
                for d in table.doras
            ]
        )
        board_info.append(" " + ["Dora:", "ドラ:"][self.language] + dora)
        uradora = "".join(
            [
                get_tile_char(d, self.is_using_unicode)
                + (
                    ""
                    if get_tile_char(d, self.is_using_unicode) == "\U0001F004\uFE0E"
                    else " "
                )
                for d in table.uradoras
            ]
        )
        if uradora != "":
            board_info.append(" " + ["UraDora:", "裏ドラ:"][self.language] + uradora)
        board_info.append("\n\n")
        board_info = "".join(board_info)
        return board_info

    def show_by_text(self, table: MahjongTable) -> str:
        self.my_idx = table.my_idx
        board_info = self.get_board_info(table)

        players_info = []
        table.players.sort(key=lambda x: (x.player_idx - self.my_idx) % 4)
        for i, p in enumerate(table.players):
            player_info = []

            player_info.append(
                get_wind_char(p.wind, self.language)
                + " [ "
                + "".join(
                    [
                        p.score
                        + (
                            [", riichi", ", リーチ"][self.language]
                            if p.is_declared_riichi
                            else ""
                        )
                    ]
                )
                + " ]",
            )
            if self.show_name:
                player_info.append(" " + p.name)
            player_info.append("\n\n")

            hand = self.get_modified_tiles(table, i, TileUnitType.HAND)
            chi = self.get_modified_tiles(table, i, TileUnitType.CHI)
            pon = self.get_modified_tiles(table, i, TileUnitType.PON)
            open_kan = self.get_modified_tiles(table, i, TileUnitType.OPEN_KAN)
            closed_kan = self.get_modified_tiles(table, i, TileUnitType.CLOSED_KAN)
            added_kan = self.get_modified_tiles(table, i, TileUnitType.ADDED_KAN)
            hand_area = hand + "      " + chi + pon + open_kan + closed_kan + added_kan
            player_info.append(hand_area)
            player_info.append("\n\n")

            discards = self.get_modified_tiles(table, i, TileUnitType.DISCARD)
            player_info.append(discards)
            player_info.append("\n\n\n")
            players_info.append("".join(player_info))
        players_info = "".join(players_info)

        system_info = []
        system_info.append(
            get_wind_char(table.players[0].wind, self.language)
            + ["'s turn now.\n", "の番です\n"][self.language]
        )
        system_info = "".join(system_info)

        return "".join([board_info, players_info, system_info, table.end_info])

    def show_by_rich(self, table: MahjongTable) -> None:

        self.my_idx = table.my_idx
        self.layout["info"].update(
            Panel(
                Text(self.get_board_info(table), justify="center", style="color(1)"),
                style="bold green",
            )
        )

        table.players.sort(key=lambda x: (x.player_idx - self.my_idx) % 4)

        players_info_top = [
            "player1_info_top",
            "player2_info_top",
            "player3_info_top",
            "player4_info_top",
        ]
        players_info_center = [
            "player1_info_center",
            "player2_info_center",
            "player3_info_center",
            "player4_info_center",
        ]
        hands_idx = ["hand1", "hand2", "hand3", "hand4"]
        discards_idx = [
            "player1_discard",
            "player2_discard",
            "player3_discard",
            "player4_discard",
        ]
        discards_idx = [
            "discard1",
            "discard2",
            "discard3",
            "discard4",
        ]

        for i, p in enumerate(table.players):
            wind = Text(
                get_wind_char(p.wind, self.language),
                justify="center",
                style="bold green",
            )

            score = Text(p.score, justify="center", style="yellow")

            riichi = Text()
            if p.is_declared_riichi:
                riichi = [
                    Text(" riichi", style="yellow"),
                    Text(" リーチ", style="yellow"),
                ][self.language]

            player_info = wind + Text("\n") + score + riichi

            self.layout[players_info_center[i]].update(player_info)

            name = Text(justify="center", style="bold green")
            if self.show_name:
                name += Text(" " + p.name, style="white")
                self.layout[players_info_top[i]].update(
                    Panel(player_info + Text("\n\n") + name, style="bold green")
                )
            hand = self.get_modified_tiles(table, i, TileUnitType.HAND)
            chi = self.get_modified_tiles(table, i, TileUnitType.CHI)
            pon = self.get_modified_tiles(table, i, TileUnitType.PON)
            open_kan = self.get_modified_tiles(table, i, TileUnitType.OPEN_KAN)
            closed_kan = self.get_modified_tiles(table, i, TileUnitType.CLOSED_KAN)
            added_kan = self.get_modified_tiles(table, i, TileUnitType.ADDED_KAN)
            if p.player_idx in [(table.my_idx + 1) % 4, (table.my_idx + 2) % 4]:
                hand_area = (
                    chi + pon + open_kan + closed_kan + added_kan + "      " + hand
                )
            else:
                hand_area = (
                    hand + "      " + chi + pon + open_kan + closed_kan + added_kan
                )
            self.layout[hands_idx[i]].update(
                Panel(
                    Text(hand_area, justify="center", no_wrap=True, style="white"),
                    style="bold green",
                )
            )

            discards = Text(
                self.get_modified_tiles(table, i, TileUnitType.DISCARD),
                justify="left",
                style="white",
            )
            self.layout[discards_idx[i]].update(Panel(discards, style="bold green"))

        console = Console()
        console.print(self.layout)
        console.print(Text(table.end_info, justify="center"))


def main():
    """
    >>> game_board = GameBoard("observations.json", "obs", False, False, 0 , True)
    >>> print(game_board.show_by_text(game_board.load_data()[0]))  # doctest: +NORMALIZE_WHITESPACE
        round:1 wall:70 Dora:sw
    <BLANKLINE>
    SOUTH [ 25000 ] target-player
    <BLANKLINE>
    m2 m6 p1 p5 p7 p8 s4 s7 ew ww ww nw nw
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    WEST [ 25000 ] rule-based-2
    <BLANKLINE>
    # # # # # # # # # # # # #
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    NORTH [ 25000 ] rule-based-3
    <BLANKLINE>
    # # # # # # # # # # # # #
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    EAST [ 25000 ] rule-based-0
    <BLANKLINE>
    # # # # # # # # # # # # #
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    SOUTH's turn now.
    <BLANKLINE>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="observations.json")
    parser.add_argument("--mode", choices=["obs", "sta"], default="obs")
    parser.add_argument("--uni", action="store_true")
    parser.add_argument("--rich", action="store_true")
    parser.add_argument("--show_name", action="store_true")
    parser.add_argument("--lang", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    game_board = GameBoard(
        args.path, args.mode, args.uni, args.rich, args.lang, args.show_name
    )

    game_data = game_board.load_data()

    turns = len(game_data)
    i = 0

    if args.rich:
        game_board.show_by_rich(game_data[i])
    else:
        print(game_board.show_by_text(game_data[i]))
    command = input("z:-20 x:-1 c:+1 v:+20 :")

    while command != "q":
        if command == "z":
            i = (i - 20) % turns
        if command == "x":
            i = (i - 1) % turns
        if command == "c":
            i = (i + 1) % turns
        if command == "v":
            i = (i + 20) % turns
        if command == "a":
            i = 0

        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

        game_data[i].check_num_tiles()

        if args.rich:
            game_board.show_by_rich(game_data[i])
        else:
            print(game_board.show_by_text(game_data[i]))

        command = input("z:-20 x:-1 c:+1 v:+20 :")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    main()
