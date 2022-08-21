from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

from google.protobuf import json_format

import mjxproto
from mjx.action import Action
from mjx.const import EventType, RelativePlayerIdx
from mjx.open import Open
from mjx.tile import Tile as _Tile
from mjx.visualizer.converter import get_event_type, get_modifier, get_tile_char, get_wind_char


@dataclass
class GameVisualConfig:
    path: str = "observations.json"
    mode: str = "obs"
    uni: bool = False
    rich: bool = False
    lang: int = 0
    show_name: bool = True


class Tile(_Tile):
    def __init__(
        self,
        tile_id: int,
        is_open: bool = False,
        is_tsumogiri: bool = False,
        with_riichi: bool = False,
        highlighting: bool = False,
    ):
        super().__init__(tile_id)
        self.visual_char: str = "error"  # 実際に表示するときの文字列（裏向き含む）
        self.is_open: bool = is_open  # 表向きか（自分の手牌以外は基本的に裏）
        self.is_tsumogiri: bool = is_tsumogiri  # ツモ切り表示にするかどうか
        self.with_riichi: bool = with_riichi  # 横向きにするかどうか
        self.is_transparent: bool = False  # 鳴かれた牌は透明にして河に表示
        self.is_highlighting: bool = highlighting  # 最後のアクションをハイライト表示


class TileUnit:
    def __init__(
        self,
        tiles_type: EventType,
        from_who: Optional[RelativePlayerIdx],
        tiles: List[Tile],
    ):
        self.tile_unit_type: EventType = tiles_type
        self.from_who: Optional[RelativePlayerIdx] = from_who
        self.tiles: List[Tile] = tiles


class Player:
    def __init__(self, idx: int):
        self.player_idx: int = idx
        self.wind: int = 0
        self.score: str = ""
        self.tile_units: List[TileUnit] = [
            TileUnit(
                EventType.DISCARD,
                None,
                [],
            )
        ]
        self.riichi_now: bool = False
        self.is_declared_riichi: bool = False
        self.name: str = ""
        self.draw_now: bool = False
        self.win_tile_id: int = -1


class MahjongTable:
    """
    MahjongTableクラスは場の情報（プレイヤーの手牌や河など）を保持します。
    """

    def __init__(self):
        self.players = [Player(i) for i in range(4)]
        self.riichi: int = 0
        self.round: int = 0
        self.honba: int = 0
        self.my_idx: int = 0  # The player you want to show.
        self.wall_num: int = 136
        self.doras: List[int] = []
        self.uradoras: List[int] = []
        self.result: str = ""
        self.event_info: Optional[EventType] = None
        self.legal_actions: List[Action] = []
        self.new_dora: Optional[int] = None
        self.latest_tile: Optional[int] = None

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
                if tile_unit.tile_unit_type == EventType.DRAW:
                    num_of_tiles += len(tile_unit.tiles)
                    hand = " ".join([str(tile.id) for tile in tile_unit.tiles])
                elif tile_unit.tile_unit_type != EventType.DISCARD:
                    num_of_tiles += 3
                    open.append(" ".join([str(tile.id) for tile in tile_unit.tiles]))

            open_str = " : ".join(open)
            if num_of_tiles < 13 or 14 < num_of_tiles:
                sys.stderr.write(
                    f"ERROR: The number of tiles is inaccurate. Player: {p.player_idx}\nhand:[{hand}],open:[{open_str}]"
                )
                return False
        return True

    @staticmethod
    def load_proto_data(path) -> List[Union[mjxproto.Observation, mjxproto.State]]:
        print(path, flush=True)
        with open(path, "r") as f:
            proto_data_list = []
            for line in f:
                line = line.strip("\n").strip()
                proto_data = MahjongTable.json_to_proto(line)
                proto_data_list.append(proto_data)
        return proto_data_list

    @classmethod
    def from_proto(cls, proto_data: Union[mjxproto.Observation, mjxproto.State]):
        if isinstance(proto_data, mjxproto.Observation):
            return cls.decode_observation(proto_data)
        else:
            return cls.decode_state(proto_data)

    @staticmethod
    def json_to_proto(json_data: str) -> Union[mjxproto.State, mjxproto.Observation]:
        try:
            observation = json_format.ParseDict(json.loads(json_data), mjxproto.Observation())
            return observation
        except json_format.ParseError:
            try:
                state = json_format.ParseDict(json.loads(json_data), mjxproto.State())
                return state
            except json_format.ParseError:
                raise ValueError(
                    f"Input json cannot be converted to either State or Observation.\n{json_data}"
                )

    @classmethod
    def decode_private_observation(
        cls, table: MahjongTable, private_observation: mjxproto.PrivateObservation, who: int
    ):
        """
        MahjongTableのデータに、
        手牌の情報を読み込ませる関数
        """
        p = table.players[who]

        has_tsumotile = False
        if private_observation.draw_history != [] and (
            private_observation.draw_history[-1]
            in [id for id in private_observation.curr_hand.closed_tiles]
        ):
            has_tsumotile = True

        if p.draw_now and has_tsumotile:
            p.tile_units.append(
                TileUnit(
                    EventType.DRAW,
                    None,
                    [
                        Tile(i, is_open=True)
                        for i in private_observation.curr_hand.closed_tiles
                        if i != private_observation.draw_history[-1]
                    ]
                    + [
                        Tile(private_observation.draw_history[-1], is_open=True, highlighting=True)
                    ],
                )
            )
            table.latest_tile = private_observation.draw_history[-1]
        else:
            p.tile_units.append(
                TileUnit(
                    EventType.DRAW,
                    None,
                    [Tile(i, is_open=True) for i in private_observation.curr_hand.closed_tiles],
                )
            )

        return table

    @classmethod
    def decode_public_observation(
        cls, table: MahjongTable, public_observation: mjxproto.PublicObservation
    ):
        """
        MahjongTableのデータに、
        手牌**以外**の情報を読み込ませる関数
        """
        table.round = public_observation.init_score.round + 1
        table.honba = public_observation.init_score.honba
        table.riichi = public_observation.init_score.riichi
        table.doras = list(public_observation.dora_indicators)

        for i, p in enumerate(table.players):
            p.name = public_observation.player_ids[i]
            p.score = str(public_observation.init_score.tens[i])

        for i, eve in enumerate(public_observation.events):
            p = table.players[eve.who]

            p.draw_now = eve.type == EventType.DRAW

            if eve.type == EventType.DISCARD:
                discard = [
                    _discard
                    for _discard in p.tile_units
                    if _discard.tile_unit_type == EventType.DISCARD
                ][0]

                if p.riichi_now:
                    discard.tiles.append(
                        Tile(
                            eve.tile,
                            is_open=True,
                            with_riichi=True,
                        )
                    )
                    p.riichi_now = False
                else:
                    discard.tiles.append(
                        Tile(
                            eve.tile,
                            is_open=True,
                        )
                    )
                if i == len(public_observation.events) - 1:
                    discard.tiles[-1].is_highlighting = True
                    table.latest_tile = discard.tiles[-1].id()

            elif eve.type == EventType.TSUMOGIRI:
                discard = [
                    _discard
                    for _discard in p.tile_units
                    if _discard.tile_unit_type == EventType.DISCARD
                ][0]

                if p.riichi_now:
                    discard.tiles.append(
                        Tile(
                            eve.tile,
                            is_open=True,
                            is_tsumogiri=True,
                            with_riichi=True,
                        )
                    )
                    p.riichi_now = False
                else:
                    discard.tiles.append(Tile(eve.tile, is_open=True, is_tsumogiri=True))

                if i == len(public_observation.events) - 1:
                    discard.tiles[-1].is_highlighting = True
                    table.latest_tile = discard.tiles[-1].id()

            elif eve.type == EventType.RIICHI:
                p.riichi_now = True

            elif eve.type == EventType.RIICHI_SCORE_CHANGE:
                table.riichi += 1
                p.is_declared_riichi = True
                p.score = str(int(p.score) - 1000)
                if i == len(public_observation.events) - 1:
                    discard = [
                        _discard
                        for _discard in p.tile_units
                        if _discard.tile_unit_type == EventType.DISCARD
                    ][0]
                    discard.tiles[-1].is_highlighting = True
                    table.latest_tile = discard.tiles[-1].id()

            elif eve.type == EventType.RON:
                if public_observation.events[i - 1].type != EventType.RON:
                    _p = table.players[public_observation.events[i - 1].who]
                    discard = [
                        _discard
                        for _discard in _p.tile_units
                        if _discard.tile_unit_type == EventType.DISCARD
                    ][0]
                    discard.tiles[-1].is_transparent = True
                    table.latest_tile = discard.tiles[-1].id()

            elif eve.type in [
                EventType.CHI,
                EventType.PON,
                EventType.CLOSED_KAN,
                EventType.ADDED_KAN,
                EventType.OPEN_KAN,
            ]:
                open_data = Open(eve.open)
                open_tiles = open_data.tiles()

                if eve.type == EventType.ADDED_KAN:
                    # added_kanのときにすでに存在するポンを取り除く処理
                    p.tile_units = [
                        t_u
                        for t_u in p.tile_units
                        if (
                            t_u.tile_unit_type != EventType.PON
                            or t_u.tiles[0].id() // 4 != open_tiles[0].id() // 4
                        )
                    ]

                # 鳴き牌を追加する処理
                if eve.type in [
                    EventType.CLOSED_KAN,
                    EventType.ADDED_KAN,
                ]:
                    p.tile_units.append(
                        TileUnit(
                            open_data.event_type(),
                            open_data.steal_from(),
                            [Tile(i.id(), is_open=True) for i in open_tiles],
                        )
                    )
                    if i == len(public_observation.events) - 1:
                        p.tile_units[-1].tiles[0].is_highlighting = True
                        table.latest_tile = p.tile_units[-1].tiles[0].id()

                # 鳴かれた牌を透明にする処理
                if eve.type in [
                    EventType.CHI,
                    EventType.PON,
                    EventType.OPEN_KAN,
                ]:
                    idx_from = p.player_idx
                    if open_data.steal_from() == RelativePlayerIdx.LEFT:
                        idx_from = (p.player_idx + 3) % 4
                    elif open_data.steal_from() == RelativePlayerIdx.CENTER:
                        idx_from = (p.player_idx + 2) % 4
                    elif open_data.steal_from() == RelativePlayerIdx.RIGHT:
                        idx_from = (p.player_idx + 1) % 4

                    p_from = table.players[idx_from]
                    for p_from_t_u in p_from.tile_units:
                        if p_from_t_u.tile_unit_type == EventType.DISCARD:
                            p_from_t_u.tiles[-1].is_transparent = True

                            # 鳴き牌を追加する処理
                            p.tile_units.append(
                                TileUnit(
                                    open_data.event_type(),
                                    open_data.steal_from(),
                                    [
                                        Tile(
                                            p_from_t_u.tiles[-1].id(),
                                            is_open=True,
                                            highlighting=(
                                                i == len(public_observation.events) - 1
                                            ),  # 最新のアクションならハイライト
                                        )
                                    ]
                                    + [
                                        Tile(i.id(), is_open=True)
                                        for i in open_tiles
                                        if i.id() != p_from_t_u.tiles[-1].id()
                                    ],
                                )
                            )
                            assert p.tile_units[-1].tiles[0].id() == p_from_t_u.tiles[-1].id()

            elif eve.type == EventType.NEW_DORA:
                table.new_dora = eve.tile

            if eve.type != EventType.NEW_DORA:
                table.new_dora = None

        if len(public_observation.events) == 0:
            return table

        last_event = public_observation.events[-1]

        if last_event.type in [
            EventType.TSUMO,
            EventType.RON,
        ]:
            table.result = "win"
            table.event_info = last_event.type
            table.players[last_event.who].win_tile_id = last_event.tile
        elif last_event.type in [
            EventType.ABORTIVE_DRAW_NINE_TERMINALS,
            EventType.ABORTIVE_DRAW_FOUR_RIICHIS,
            EventType.ABORTIVE_DRAW_THREE_RONS,
            EventType.ABORTIVE_DRAW_FOUR_KANS,
            EventType.ABORTIVE_DRAW_FOUR_WINDS,
            EventType.ABORTIVE_DRAW_NORMAL,
            EventType.ABORTIVE_DRAW_NAGASHI_MANGAN,
        ]:
            table.result = "nowinner"
            table.event_info = last_event.type
            discard = [
                _discard
                for _discard in table.players[public_observation.events[-2].who].tile_units
                if _discard.tile_unit_type == EventType.DISCARD
            ][0]
            if len(discard.tiles) > 0:
                discard.tiles[-1].is_highlighting = True
                table.latest_tile = discard.tiles[-1].id()

        return table

    @classmethod
    def decode_round_terminal(cls, table: MahjongTable, round_terminal, win: bool):
        final_ten_changes = [0, 0, 0, 0]
        if win:
            for win_data in round_terminal.wins:
                table.uradoras = win_data.ura_dora_indicators
                for i in range(4):
                    p = table.players[i]
                    if p.player_idx == win_data.who:
                        # 手牌をround_terminalのもので上書き
                        p.tile_units = [
                            t_u for t_u in p.tile_units if t_u.tile_unit_type != EventType.DRAW
                        ]

                        p.tile_units.append(
                            TileUnit(
                                EventType.DRAW,
                                None,
                                [
                                    Tile(
                                        i,
                                        is_open=True,
                                        highlighting=(i == p.win_tile_id),
                                    )
                                    for i in win_data.hand.closed_tiles
                                ],
                            )
                        )

                    final_ten_changes[i] += win_data.ten_changes[i]

            # 複数人勝者が居た場合のために、後でまとめて点数移動を計算
            for i, p in enumerate(table.players):
                delta = final_ten_changes[i]
                p.score = (
                    str(int(p.score) + delta) + ("(+" if delta > 0 else "(") + str(delta) + ")"
                )

        else:  # nowinner
            for tenpai_data in round_terminal.no_winner.tenpais:
                for i in range(4):
                    if table.players[i].player_idx == tenpai_data.who:
                        # 手牌をround_terminalのもので上書き
                        table.players[i].tile_units = [
                            t_u
                            for t_u in table.players[i].tile_units
                            if t_u.tile_unit_type != EventType.DRAW
                        ]

                        table.players[i].tile_units.append(
                            TileUnit(
                                EventType.DRAW,
                                None,
                                [Tile(i, is_open=True) for i in tenpai_data.hand.closed_tiles],
                            )
                        )

            for i, p in enumerate(table.players):
                delta = round_terminal.no_winner.ten_changes[i]
                p.score = (
                    str(int(p.score) + delta) + ("(+" if delta > 0 else "(") + str(delta) + ")"
                )

        return table

    @classmethod
    def decode_observation(cls, proto_data: mjxproto.Observation):
        table = MahjongTable()

        table.my_idx = proto_data.who
        table = cls.decode_public_observation(table, proto_data.public_observation)
        table = cls.decode_private_observation(
            table, proto_data.private_observation, proto_data.who
        )

        # Obsevationの場合,適切な数の裏向きの手牌を用意する
        for i, p in enumerate(table.players):
            if p.player_idx != table.my_idx:
                hands_num = 13
                for t_u in p.tile_units:
                    if t_u.tile_unit_type not in [
                        EventType.DISCARD,
                        EventType.DRAW,
                    ]:
                        hands_num -= 3

                p.tile_units.append(
                    TileUnit(
                        EventType.DRAW,
                        None,
                        [Tile(i, is_open=False) for i in range(hands_num)],
                    )
                )
            p.wind = (-table.round + 1 + i) % 4

        table.wall_num = cls.get_wall_num(table)
        table.legal_actions = [Action.from_proto(act) for act in proto_data.legal_actions]
        table.legal_actions.sort(key=lambda a: a.to_idx())

        if len(proto_data.public_observation.events) != 0:
            if table.result == "win":
                table = cls.decode_round_terminal(table, proto_data.round_terminal, True)

            elif table.result == "nowinner":
                table = cls.decode_round_terminal(table, proto_data.round_terminal, False)

        return table

    @classmethod
    def decode_state(cls, proto_data: mjxproto.State):
        table = MahjongTable()

        table = cls.decode_public_observation(table, proto_data.public_observation)
        for i in range(4):
            table = cls.decode_private_observation(table, proto_data.private_observations[i], i)

        for i, p in enumerate(table.players):
            p.wind = (-table.round + 1 + i) % 4

        table.wall_num = cls.get_wall_num(table)

        if len(proto_data.public_observation.events) != 0:
            if table.result == "win":
                table = cls.decode_round_terminal(table, proto_data.round_terminal, True)

            elif table.result == "nowinner":
                table = cls.decode_round_terminal(table, proto_data.round_terminal, False)

        return table


class GameBoardVisualizer:
    """
    GameBoardVisualizer クラスは内部にMahjongTableクラスのオブジェクトを持ち、
    その表示などを行います。
    """

    def __init__(self, config: GameVisualConfig):
        self.config = config

    def get_layout(self):
        from rich.layout import Layout

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
                tile.visual_char = "\U0001F02B" if self.config.uni else "#"
            else:
                tile.visual_char = get_tile_char(tile.id(), self.config.uni)
            if tile.is_tsumogiri:
                if tile.with_riichi:
                    if self.config.uni and tile.visual_char != "\U0001F004\uFE0E":
                        tile.visual_char += " *r"
                    else:
                        tile.visual_char += "*r"
                else:
                    if self.config.uni and tile.visual_char != "\U0001F004\uFE0E":
                        tile.visual_char += " *"
                    else:
                        tile.visual_char += "*"
            elif tile.with_riichi:
                if self.config.uni and tile.visual_char != "\U0001F004\uFE0E":
                    tile.visual_char += " r"
                else:
                    tile.visual_char += "r"

        if self.config.rich:
            if tile_unit.tile_unit_type == EventType.DISCARD:
                discards = [
                    tile.visual_char
                    + (
                        ""
                        if (
                            tile.visual_char == "\U0001F004\uFE0E"
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
                            (" " if tile.visual_char == "\U0001F004\uFE0E" else "")
                            + tile.visual_char
                            for tile in tile_unit.tiles
                        ]
                    )
                    + "\n"
                )
                return tiles
            elif player_idx == 2:
                if tile_unit.tile_unit_type == EventType.DRAW:
                    tiles = "".join(
                        [
                            tile.visual_char
                            + ("" if tile.visual_char == "\U0001F004\uFE0E" else " ")
                            for tile in tile_unit.tiles
                        ]
                    )
                    return tiles
                tiles = get_modifier(tile_unit.from_who, tile_unit.tile_unit_type) + "".join(
                    [
                        tile.visual_char
                        + (
                            ""
                            if (not self.config.uni or tile.visual_char == "\U0001F004\uFE0E")
                            else " "
                        )
                        for tile in sorted(
                            tile_unit.tiles,
                            key=lambda x: x.id(),
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
                            (" " if tile.visual_char == "\U0001F004\uFE0E" else "")
                            + tile.visual_char
                            for tile in tile_unit.tiles
                        ]
                    )
                    + "\n "
                    + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                    + "\n"
                )
                return tiles
            else:
                if tile_unit.tile_unit_type == EventType.DRAW:
                    tiles = "".join(
                        [
                            tile.visual_char
                            + ("" if tile.visual_char == "\U0001F004\uFE0E" else " ")
                            for tile in tile_unit.tiles
                        ]
                    )
                    return tiles
                tiles = (
                    "".join(
                        [
                            tile.visual_char
                            + (
                                ""
                                if (not self.config.uni or tile.visual_char == "\U0001F004\uFE0E")
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
            if tile_unit.tile_unit_type == EventType.DRAW:
                tiles = "".join(
                    [
                        tile.visual_char + ("" if tile.visual_char == "\U0001F004\uFE0E" else " ")
                        for tile in tile_unit.tiles
                    ]
                )
                return tiles
            if tile_unit.tile_unit_type == EventType.DISCARD:
                discards = [
                    tile.visual_char
                    + (
                        ""
                        if (
                            tile.visual_char == "\U0001F004\uFE0E"
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
                EventType.CHI,
                EventType.PON,
                EventType.CLOSED_KAN,
                EventType.OPEN_KAN,
                EventType.ADDED_KAN,
            ]:
                tiles = "".join(
                    [
                        tile.visual_char
                        + (
                            ""
                            if (not self.config.uni or tile.visual_char == "\U0001F004\uFE0E")
                            else " "
                        )
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

        if table.event_info is not None:
            event_info = get_event_type(table.event_info, self.config.lang)
            board_info.append("    " + event_info)

        return "".join(board_info)

    def get_text_width(self, text: str):
        import unicodedata

        text_counter: int = 0
        for c in text:
            if unicodedata.east_asian_width(c) in "FWA":
                text_counter = text_counter + 2
            else:
                text_counter = text_counter + 1
        return text_counter

    def show_by_text(self, table: MahjongTable) -> str:
        board_info = self.get_board_info(table)
        board_info = (
            "#" * (self.get_text_width(board_info) + 2)
            + "\n"
            + "#"
            + board_info
            + "#\n"
            + "#" * (self.get_text_width(board_info) + 2)
            + "\n"
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
            hand = ""
            discards = ""
            for t_u in reversed(p.tile_units):
                if t_u.tile_unit_type == EventType.DRAW:
                    hand = self.add_suffix(t_u)
                elif t_u.tile_unit_type == EventType.DISCARD:
                    discards = self.add_suffix(t_u)
                else:
                    opens.append(self.add_suffix(t_u))

            hand_area = hand + "      " + " ".join(opens)
            player_info.append(hand_area)
            player_info.append("\n\n")

            player_info.append(discards)
            player_info.append("\n\n\n")
            players_info.append("".join(player_info))

        return board_info + "".join(players_info)

    def show_by_rich(self, table: MahjongTable) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

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
            hand = ""
            discards = ""
            for t_u in reversed(p.tile_units):
                if t_u.tile_unit_type == EventType.DRAW:
                    hand = self.add_suffix(t_u, player_idx=i)
                elif t_u.tile_unit_type == EventType.DISCARD:
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

            layout[discards_idx[i]].update(
                Panel(
                    Text(
                        discards,
                        justify="left",
                        style="white",
                    ),
                    style="bold green",
                )
            )

        console = Console()
        console.print(layout)

    def print(self, data: MahjongTable):
        if self.config.rich:
            self.show_by_rich(data)
        else:
            print(self.show_by_text(data))
