import json
import sys
from dataclasses import dataclass

import mjxproto
from google.protobuf import json_format
from mjx.visualizer.converter import (
    FromWho,
    TileUnitType,
    get_event_type,
    get_modifier,
    get_tile_char,
    get_wind_char,
)
from mjxproto import EventType
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from .open_utils import open_event_type, open_from, open_tile_ids


@dataclass
class GameVisualConfig:
    path: str = "observations.json"
    mode: str = "obs"
    uni: bool = False
    rich: bool = False
    lang: int = 0
    show_name: bool = True


class Tile:
    def __init__(
        self,
        tile_id: int,
        is_open: bool = False,
        is_tsumogiri: bool = False,
        with_riichi: bool = False,
    ):
        self.id = tile_id
        self.is_open = is_open
        self.is_tsumogiri = is_tsumogiri
        self.with_riichi = with_riichi
        self.is_transparent = False  # 鳴かれた牌は透明にして河に表示


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
    def __init__(self, idx: int):
        self.player_idx = idx
        self.wind = 0
        self.score = ""
        self.tile_units = [
            TileUnit(
                TileUnitType.DISCARD,
                FromWho.NONE,
                [],
            )
        ]
        self.riichi_now = False
        self.is_declared_riichi = False
        self.name = ""
        self.draw_now = False


class MahjongTable:
    """
    MahjongTableクラスは場の情報（プレイヤーの手牌や河など）を保持します。
    """

    def __init__(self):
        self.players = [Player(i) for i in range(4)]
        self.riichi = 0
        self.round = 0
        self.honba = 0
        self.my_idx = 0  # 0-3; The player you want to show.
        self.wall_num = 134
        self.doras = []
        self.uradoras = []
        self.result = ""
        self.event_info = ""

    def get_wall_num(self) -> int:
        all = 136 - 14
        for p in self.players:
            for t_u in p.tile_units:
                all -= len([tile for tile in t_u.tiles if not tile.is_transparent])
        return all

    def check_num_tiles(self) -> bool:
        for p in self.players:
            num_of_tiles = 0
            hand = ""
            open = []
            for tile_unit in p.tile_units:
                if tile_unit.tile_unit_type == TileUnitType.HAND:
                    num_of_tiles += len(tile_unit.tiles)
                    hand = " ".join([str(tile.id) for tile in tile_unit.tiles])
                elif tile_unit.tile_unit_type != TileUnitType.DISCARD:
                    num_of_tiles += 3
                    open.append(" ".join([str(tile.id) for tile in tile_unit.tiles]))

            open = " : ".join(open)
            if num_of_tiles < 13 or 14 < num_of_tiles:
                sys.stderr.write(
                    f"ERROR: The number of tiles is inaccurate. Player: {p.player_idx}\nhand:[{hand}],open:[{open}]"
                )
                return False
        return True

    @classmethod
    def load_data(cls, path, mode) -> list:
        with open(path, "r", errors="ignore") as f:
            tables = []
            for line in f:
                if mode == "obs":
                    table = cls.decode_observation(line)
                else:
                    table = cls.decode_state(line)
                assert table.check_num_tiles()
                tables.append(table)
        return tables

    @classmethod
    def decode_private_observation(cls, table, private_observation, who: int):
        """
        MahjongTableのデータに、
        手牌の情報を読み込ませる関数
        """
        has_tsumotile = False
        if private_observation.draw_history != [] and (
            private_observation.draw_history[-1]
            in [id for id in private_observation.curr_hand.closed_tiles]
        ):
            has_tsumotile = True

        if table.players[who].draw_now and has_tsumotile:
            table.players[who].tile_units.append(
                TileUnit(
                    TileUnitType.HAND,
                    FromWho.NONE,
                    [
                        Tile(i, is_open=True)
                        for i in private_observation.curr_hand.closed_tiles
                        if i != private_observation.draw_history[-1]
                    ]
                    + [Tile(private_observation.draw_history[-1], is_open=True)],
                )
            )
        else:
            table.players[who].tile_units.append(
                TileUnit(
                    TileUnitType.HAND,
                    FromWho.NONE,
                    [Tile(i, is_open=True) for i in private_observation.curr_hand.closed_tiles],
                )
            )

        return table

    @classmethod
    def decode_public_observation(cls, table, public_observation):
        """
        MahjongTableのデータに、
        手牌**以外**の情報を読み込ませる関数
        """
        table.round = public_observation.init_score.round + 1
        table.honba = public_observation.init_score.honba
        table.riichi = public_observation.init_score.riichi
        table.doras = public_observation.dora_indicators

        for i, p in enumerate(table.players):
            p.name = public_observation.player_ids[i]
            p.score = str(public_observation.init_score.tens[i])

        for i, eve in enumerate(public_observation.events):
            p = table.players[eve.who]

            p.draw_now = eve.type == EventType.EVENT_TYPE_DRAW

            if eve.type == EventType.EVENT_TYPE_DISCARD:
                for t_u in p.tile_units:
                    if t_u.tile_unit_type == TileUnitType.DISCARD:
                        if p.riichi_now:
                            t_u.tiles.append(
                                Tile(
                                    eve.tile,
                                    is_open=True,
                                    with_riichi=True,
                                )
                            )
                            p.riichi_now = False
                        else:
                            t_u.tiles.append(
                                Tile(
                                    eve.tile,
                                    True,
                                )
                            )

            if eve.type == EventType.EVENT_TYPE_TSUMOGIRI:
                for t_u in p.tile_units:
                    if t_u.tile_unit_type == TileUnitType.DISCARD:
                        if p.riichi_now:
                            t_u.tiles.append(
                                Tile(
                                    eve.tile,
                                    is_open=True,
                                    is_tsumogiri=True,
                                    with_riichi=True,
                                )
                            )
                            p.riichi_now = False
                        else:
                            t_u.tiles.append(Tile(eve.tile, is_open=True, is_tsumogiri=True))

            if eve.type == EventType.EVENT_TYPE_RIICHI:
                p.riichi_now = True

            if eve.type == EventType.EVENT_TYPE_RIICHI_SCORE_CHANGE:
                table.riichi += 1
                p.is_declared_riichi = True
                p.score = str(int(p.score) - 1000)

            if eve.type == EventType.EVENT_TYPE_RON:
                if public_observation.events[i - 1].type != EventType.EVENT_TYPE_RON:
                    p = table.players[public_observation.events[i - 1].who]
                    for t_u in p.tile_units:
                        if t_u.tile_unit_type == TileUnitType.DISCARD:
                            t_u.tiles[-1].is_transparent = True

            if eve.type in [
                EventType.EVENT_TYPE_CHI,
                EventType.EVENT_TYPE_PON,
                EventType.EVENT_TYPE_CLOSED_KAN,
                EventType.EVENT_TYPE_ADDED_KAN,
                EventType.EVENT_TYPE_OPEN_KAN,
            ]:
                if eve.type == EventType.EVENT_TYPE_ADDED_KAN:
                    # added_kanのときにすでに存在するポンを取り除く処理
                    p.tile_units = [
                        t_u
                        for t_u in p.tile_units
                        if (
                            t_u.tile_unit_type != TileUnitType.PON
                            or t_u.tiles[0].id // 4 != open_tile_ids(eve.open)[0] // 4
                        )
                    ]

                # 鳴き牌を追加する処理
                if eve.type in [
                    EventType.EVENT_TYPE_CLOSED_KAN,
                    EventType.EVENT_TYPE_ADDED_KAN,
                ]:
                    p.tile_units.append(
                        TileUnit(
                            open_event_type(eve.open),
                            open_from(eve.open),
                            [Tile(i, is_open=True) for i in open_tile_ids(eve.open)],
                        )
                    )

                # 鳴かれた牌を透明にする処理
                if eve.type in [
                    EventType.EVENT_TYPE_CHI,
                    EventType.EVENT_TYPE_PON,
                    EventType.EVENT_TYPE_OPEN_KAN,
                ]:
                    idx_from = p.player_idx
                    if open_from(eve.open) == FromWho.LEFT:
                        idx_from = (p.player_idx + 3) % 4
                    elif open_from(eve.open) == FromWho.MID:
                        idx_from = (p.player_idx + 2) % 4
                    elif open_from(eve.open) == FromWho.RIGHT:
                        idx_from = (p.player_idx + 1) % 4

                    p_from = table.players[idx_from]
                    for p_from_t_u in p_from.tile_units:
                        if p_from_t_u.tile_unit_type == TileUnitType.DISCARD:
                            p_from_t_u.tiles[-1].is_transparent = True

                            # 鳴き牌を追加する処理
                            p.tile_units.append(
                                TileUnit(
                                    open_event_type(eve.open),
                                    open_from(eve.open),
                                    [Tile(p_from_t_u.tiles[-1].id, is_open=True)]
                                    + [
                                        Tile(i, is_open=True)
                                        for i in open_tile_ids(eve.open)
                                        if i != p_from_t_u.tiles[-1].id
                                    ],
                                )
                            )
                            assert p.tile_units[-1].tiles[0].id == p_from_t_u.tiles[-1].id

        if len(public_observation.events) == 0:
            return table

        if public_observation.events[-1].type in [
            EventType.EVENT_TYPE_TSUMO,
            EventType.EVENT_TYPE_RON,
        ]:
            table.result = "win"
            table.event_info = public_observation.events[-1].type
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
            table.event_info = public_observation.events[-1].type

        return table

    @classmethod
    def decode_round_terminal(cls, table, round_terminal, win: bool):
        final_ten_changes = [0, 0, 0, 0]
        if win:
            for win_data in round_terminal.wins:
                table.uradoras = win_data.ura_dora_indicators
                for i in range(4):
                    final_ten_changes[i] += win_data.ten_changes[i]

            for i, p in enumerate(table.players):
                delta = final_ten_changes[i]
                p.score = (
                    str(int(p.score) + delta) + ("(+" if delta > 0 else "(") + str(delta) + ")"
                )
        else:
            for i, p in enumerate(table.players):
                delta = round_terminal.no_winner.ten_changes[i]
                p.score = (
                    str(int(p.score) + delta) + ("(+" if delta > 0 else "(") + str(delta) + ")"
                )

        return table

    @classmethod
    def decode_observation(cls, jsondata):
        d = json.loads(jsondata)
        gamedata = json_format.ParseDict(d, mjxproto.Observation())

        table = MahjongTable()

        table.my_idx = gamedata.who
        table = cls.decode_public_observation(table, gamedata.public_observation)
        table = cls.decode_private_observation(table, gamedata.private_observation, gamedata.who)

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
                        [Tile(i, is_open=False) for i in range(hands_num)],
                    )
                )
            p.wind = (-table.round + 1 + i) % 4

        table.wall_num = cls.get_wall_num(table)
        if len(gamedata.public_observation.events) != 0:
            if table.result == "win":
                table = cls.decode_round_terminal(table, gamedata.round_terminal, True)

            elif table.result == "nowinner":
                table = cls.decode_round_terminal(table, gamedata.round_terminal, False)

        return table

    @classmethod
    def decode_state(cls, jsondata):
        d = json.loads(jsondata)
        gamedata = json_format.ParseDict(d, mjxproto.State())

        table = MahjongTable()

        table = cls.decode_public_observation(table, gamedata.public_observation)
        for i in range(4):
            table = cls.decode_private_observation(table, gamedata.private_observations[i], i)

        for i, p in enumerate(table.players):
            p.wind = (-table.round + 1 + i) % 4

        table.wall_num = cls.get_wall_num(table)

        if len(gamedata.public_observation.events) != 0:
            if table.result == "win":
                table = cls.decode_round_terminal(table, gamedata.round_terminal, True)

            elif table.result == "nowinner":
                table = cls.decode_round_terminal(table, gamedata.round_terminal, False)

        return table


class GameBoardVisualizer:
    """
    GameBoardVisualizer クラスは内部にMahjongTableクラスのオブジェクトを持ち、
    その表示などを行います。
    """

    def __init__(self, config: GameVisualConfig):
        self.config = config

    def get_layout(self):
        layout = Layout()
        if self.config.show_name:
            layout.split_column(
                Layout(" ", name="space_top"),
                Layout(name="info"),
                Layout(name="players_info_top"),
                Layout(name="table"),
            )
            layout["players_info_top"].size = 7
            layout["players_info_top"].split_row(
                Layout(" ", name="player1_info_top"),
                Layout(" ", name="player2_info_top"),
                Layout(" ", name="player3_info_top"),
                Layout(" ", name="player4_info_top"),
            )
        else:
            layout.split_column(
                Layout(" ", name="space_top"),
                Layout(name="info"),
                Layout(name="table"),
            )

        layout["space_top"].size = 3
        layout["info"].size = 3
        layout["table"].minimum_size = 20

        layout["table"].split_column(
            Layout(name="upper1"),
            Layout(name="middle1"),
            Layout(name="lower1"),
        )

        layout["upper1"].size = 3
        layout["lower1"].size = 3

        layout["upper1"].split_row(
            Layout(" "),
            Layout(" ", name="hand3"),
            Layout(" "),
        )
        layout["hand3"].ratio = 6

        layout["middle1"].split_row(
            Layout(" ", name="hand4"),
            Layout(name="middle2"),
            Layout(" ", name="hand2"),
        )
        layout["middle2"].ratio = 10

        layout["lower1"].split_row(
            Layout(" "),
            Layout(" ", name="hand1"),
            Layout(" "),
        )
        layout["hand1"].ratio = 6

        layout["middle2"].split_column(
            Layout(name="upper2"),
            Layout(name="middle3"),
            Layout(name="lower2"),
        )

        layout["upper2"].split_row(
            Layout(" ", name="player3_info_corner"),
            Layout(" ", name="discard3"),
            Layout(" ", name="player2_info_corner"),
        )

        layout["middle3"].split_row(
            Layout(" ", name="discard4"),
            Layout(" ", name="middle4"),
            Layout(" ", name="discard2"),
        )

        layout["middle4"].split_column(
            Layout(" ", name="space_for_info_center"),
            Layout(" ", name="player3_info_center"),
            Layout(" ", name="middle5"),
            Layout(" ", name="player1_info_center"),
        )
        layout["space_for_info_center"].size = 1

        layout["middle5"].split_row(
            Layout(" ", name="player4_info_center"),
            Layout(" ", name="player2_info_center"),
        )
        layout["lower2"].split_row(
            Layout(" ", name="player4_info_corner"),
            Layout(" ", name="discard1"),
            Layout(" ", name="player1_info_corner"),
        )

        return layout

    def add_suffix(self, tile_unit: TileUnit, player_idx: int = 0) -> str:
        """
        wdwdwd -> wdwdwdL
        のように、情報を付け加える関数。
        """

        for tile in tile_unit.tiles:
            if not tile.is_open:
                tile.char = "\U0001F02B" if self.config.uni else "#"
            else:
                tile.char = get_tile_char(tile.id, self.config.uni)
            if tile.is_tsumogiri:
                if tile.with_riichi:
                    if self.config.uni and tile.char != "\U0001F004\uFE0E":
                        tile.char += " *r"
                    else:
                        tile.char += "*r"
                else:
                    if self.config.uni and tile.char != "\U0001F004\uFE0E":
                        tile.char += " *"
                    else:
                        tile.char += "*"
            elif tile.with_riichi:
                if self.config.uni and tile.char != "\U0001F004\uFE0E":
                    tile.char += " r"
                else:
                    tile.char += "r"

        if self.config.rich:
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
                        if self.config.uni
                        else " "
                    )
                    for tile in tile_unit.tiles
                ]
                tiles = "\n".join(
                    ["".join(discards[idx : idx + 6]) for idx in range(0, len(discards), 6)]
                )
                return tiles
            elif player_idx == 1:
                tiles = (
                    "\n"
                    + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                    + "\n"
                    + "\n".join(
                        [
                            (" " if tile.char == "\U0001F004\uFE0E" else "") + tile.char
                            for tile in tile_unit.tiles
                        ]
                    )
                    + "\n"
                )
                return tiles
            elif player_idx == 2:
                if tile_unit.tile_unit_type == TileUnitType.HAND:
                    tiles = "".join(
                        [
                            tile.char + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                            for tile in tile_unit.tiles
                        ]
                    )
                    return tiles
                tiles = get_modifier(tile_unit.from_who, tile_unit.tile_unit_type) + "".join(
                    [
                        tile.char
                        + ("" if (not self.config.uni or tile.char == "\U0001F004\uFE0E") else " ")
                        for tile in sorted(
                            tile_unit.tiles,
                            key=lambda x: x.id,
                            reverse=True,
                        )
                    ]
                )
                return tiles
            elif player_idx == 3:
                tiles = (
                    "\n"
                    + "\n".join(
                        [
                            (" " if tile.char == "\U0001F004\uFE0E" else "") + tile.char
                            for tile in tile_unit.tiles
                        ]
                    )
                    + "\n "
                    + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                    + "\n"
                )
                return tiles
            else:
                if tile_unit.tile_unit_type == TileUnitType.HAND:
                    tiles = "".join(
                        [
                            tile.char + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                            for tile in tile_unit.tiles
                        ]
                    )
                    return tiles
                tiles = (
                    "".join(
                        [
                            tile.char
                            + (
                                ""
                                if (not self.config.uni or tile.char == "\U0001F004\uFE0E")
                                else " "
                            )
                            for tile in tile_unit.tiles
                        ]
                    )
                    + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                    + " "
                )
                return tiles
        else:  # not rich
            if tile_unit.tile_unit_type == TileUnitType.HAND:
                tiles = "".join(
                    [
                        tile.char + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                        for tile in tile_unit.tiles
                    ]
                )
                return tiles
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
                        if self.config.uni
                        else " "
                    )
                    for tile in tile_unit.tiles
                ]
                tiles = "\n".join(
                    ["".join(discards[idx : idx + 6]) for idx in range(0, len(discards), 6)]
                )
                return tiles

            if tile_unit.tile_unit_type in [
                TileUnitType.CHI,
                TileUnitType.PON,
                TileUnitType.CLOSED_KAN,
                TileUnitType.OPEN_KAN,
                TileUnitType.ADDED_KAN,
            ]:
                tiles = "".join(
                    [
                        tile.char
                        + ("" if (not self.config.uni or tile.char == "\U0001F004\uFE0E") else " ")
                        for tile in tile_unit.tiles
                    ]
                ) + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                return tiles
        return "error"

    def get_board_info(self, table: MahjongTable) -> str:
        board_info = []
        board_info.append(
            [
                f"round:{table.round}",
                get_wind_char((table.round - 1) // 4, self.config.lang)
                + str((table.round - 1) % 4 + 1)
                + "局",
            ][self.config.lang]
        )
        if table.honba > 0:
            board_info.append(
                " " + ["honba:" + str(table.honba), str(table.honba) + "本場"][self.config.lang]
            )
        if table.riichi > 0:
            board_info.append(" " + ["riichi:", "供託"][self.config.lang] + str(table.riichi))
        board_info.append(
            " "
            + ["wall:" + str(table.wall_num), "残り:" + str(table.wall_num) + "枚"][self.config.lang]
        )
        dora = "".join(
            [
                get_tile_char(d, self.config.uni)
                + ("" if get_tile_char(d, self.config.uni) == "\U0001F004\uFE0E" else " ")
                for d in table.doras
            ]
        )
        board_info.append(" " + ["Dora:", "ドラ:"][self.config.lang] + dora)
        uradora = "".join(
            [
                get_tile_char(d, self.config.uni)
                + ("" if get_tile_char(d, self.config.uni) == "\U0001F004\uFE0E" else " ")
                for d in table.uradoras
            ]
        )
        if uradora != "":
            board_info.append(" " + ["UraDora:", "裏ドラ:"][self.config.lang] + uradora)

        event_info = get_event_type(table.event_info, self.config.lang)
        board_info.append("    " + event_info)
        board_info = "".join(board_info)
        return board_info

    def show_by_text(self, table: MahjongTable) -> str:
        board_info = self.get_board_info(table)
        board_info = (
            "#"
            + "#" * len(board_info)
            + "#\n"
            + "#"
            + board_info
            + "#\n"
            + "#"
            + "#" * len(board_info)
            + "#\n"
        )

        players_info = []
        table.players.sort(key=lambda x: (x.player_idx - table.my_idx) % 4)
        for i, p in enumerate(table.players):
            player_info = []

            player_info.append(
                get_wind_char(p.wind, self.config.lang)
                + " [ "
                + "".join(
                    [
                        p.score
                        + ([", riichi", ", リーチ"][self.config.lang] if p.is_declared_riichi else "")
                    ]
                )
                + " ]",
            )
            if self.config.show_name:
                player_info.append(" " + p.name)
            player_info.append("\n\n")

            opens = []
            for t_u in reversed(p.tile_units):
                if t_u.tile_unit_type == TileUnitType.HAND:
                    hand = self.add_suffix(t_u)
                elif t_u.tile_unit_type == TileUnitType.DISCARD:
                    discards = self.add_suffix(t_u)
                else:
                    opens.append(self.add_suffix(t_u))

            hand_area = hand + "      " + " ".join(opens)
            player_info.append(hand_area)
            player_info.append("\n\n")

            player_info.append(discards)
            player_info.append("\n\n\n")
            players_info.append("".join(player_info))
        players_info = "".join(players_info)

        system_info = []
        system_info.append(
            get_wind_char(table.players[0].wind, self.config.lang)
            + ["'s turn now.\n", "の番です\n"][self.config.lang]
        )
        system_info = "".join(system_info)

        return "".join([board_info, players_info, system_info])

    def show_by_rich(self, table: MahjongTable) -> None:
        layout = self.get_layout()

        layout["info"].update(
            Panel(
                Text(self.get_board_info(table), justify="center", style="color(1)"),
                style="bold green",
            )
        )

        table.players.sort(key=lambda x: (x.player_idx - table.my_idx) % 4)

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
                get_wind_char(p.wind, self.config.lang),
                justify="center",
                style="bold green",
            )

            score = Text(p.score, justify="center", style="yellow")

            riichi = Text()
            if p.is_declared_riichi:
                riichi = [
                    Text(" riichi", style="yellow"),
                    Text(" リーチ", style="yellow"),
                ][self.config.lang]

            player_info = wind + Text("\n") + score + riichi

            layout[players_info_center[i]].update(player_info)

            name = Text(justify="center", style="bold green")
            if self.config.show_name:
                name += Text(" " + p.name, style="white")
                layout[players_info_top[i]].update(
                    Panel(player_info + Text("\n\n") + name, style="bold green")
                )

            opens = []
            for t_u in reversed(p.tile_units):
                if t_u.tile_unit_type == TileUnitType.HAND:
                    hand = self.add_suffix(t_u, player_idx=i)
                elif t_u.tile_unit_type == TileUnitType.DISCARD:
                    discards = self.add_suffix(t_u, player_idx=i)
                else:
                    opens.append(self.add_suffix(t_u, player_idx=i))
            if p.player_idx in [(table.my_idx + 1) % 4, (table.my_idx + 2) % 4]:
                hand_area = " ".join(opens) + "      " + hand
            else:
                hand_area = hand + "      " + " ".join(opens)
            layout[hands_idx[i]].update(
                Panel(
                    Text(hand_area, justify="center", no_wrap=True, style="white"),
                    style="bold green",
                )
            )

            discards = Text(
                discards,
                justify="left",
                style="white",
            )
            layout[discards_idx[i]].update(Panel(discards, style="bold green"))

        console = Console()
        console.print(layout)

    def print(self, data: MahjongTable):
        if self.config.rich:
            self.show_by_rich(data)
        else:
            print(self.show_by_text(data))
