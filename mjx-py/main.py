import argparse
import os
import mjxproto
from converter import TileUnitType, FromWho
from converter import get_modifier, get_tile_char, get_wind_char
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
    ):
        self.id = tile_id
        self.is_tsumogiri = is_tsumogiri
        if not is_open:
            self.char = "\U0001F02B" if is_using_unicode else "#"
        else:
            self.char = get_tile_char(tile_id, is_using_unicode)
        if is_tsumogiri:
            if is_using_unicode and self.char != "\U0001F004\uFE0E":
                self.char += " *"
            else:
                self.char += "*"


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
        self.score: int
        self.tile_units = []
        self.is_declared_riichi = False
        self.name: str


class MahjongTable:
    """
    MahjongTableクラスは場の情報（プレイヤーの手牌や河など）を保持します。
    """

    def __init__(
        self,
        player1: Player,
        player2: Player,
        player3: Player,
        player4: Player,
    ):
        self.players = [player1, player2, player3, player4]
        self.riichi = 0
        self.round = 0
        self.honba = 0
        self.last_action = 0  # 0-10
        self.last_player = 0  # 0-3
        self.my_idx = 0  # 0-3; The player you want to show.
        self.wall_num = 134

    def check_num_tiles(self) -> bool:
        for p in self.players:
            for tile_unit in p.tile_units:
                if tile_unit.tile_unit_type == TileUnitType.HAND:
                    num_of_tiles = len(tile_unit.tiles)
                elif tile_unit.tile_unit_type != TileUnitType.DISCARD:
                    num_of_tiles += 3

            if num_of_tiles < 13 or 14 < num_of_tiles:
                print("ERROR: The number of tiles is inaccurate.")
                print("player", p.player_idx, ":", num_of_tiles)
                # return False
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
        self.name = mode
        self.is_using_unicode = is_using_unicode
        self.is_using_rich = is_using_rich
        self.language = language
        self.show_name = show_name
        self.my_idx = 0
        self.tables = []

        self.layout = Layout()
        self.layout.split_column(
            Layout(name="info"),
            Layout(name="players_info_top"),
            Layout(" ", name="space"),
            Layout(name="table"),
        )
        self.layout["info"].size = 3
        self.layout["players_info_top"].size = 7 if show_name else 4
        self.layout["space"].size = 1
        self.layout["table"].minimum_size = 20

        self.layout["players_info_top"].split_row(
            Layout(" ", name="player1_info_top"),
            Layout(" ", name="player2_info_top"),
            Layout(" ", name="player3_info_top"),
            Layout(" ", name="player4_info_top"),
        )

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
        self.layout["middle2"].ratio = 6

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

    def load_data(self) -> MahjongTable:
        with open(self.path, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < 10:
                    continue
                gamedata = mjxproto.Observation()
                gamedata.from_json(line)

                gamedata.private_observation.init_hand

                players = [Player(), Player(), Player(), Player()]

                for i in range(4):
                    players[i].name = gamedata.public_observation.player_ids[i]
                    players[i].score = gamedata.public_observation.init_score.tens[i]
                    players[i].player_idx = i
                    players[i].wind = i

                    if i == gamedata.who:
                        p1_hands = TileUnit(
                            TileUnitType.HAND,
                            FromWho.NONE,
                            [
                                Tile(i, True, self.is_using_unicode)
                                for i in gamedata.private_observation.curr_hand.closed_tiles
                            ],
                        )
                        players[i].tile_units.append(p1_hands)
                    else:

                        p1_hands = TileUnit(
                            TileUnitType.HAND,
                            FromWho.NONE,
                            [
                                Tile((120 + i * 4) % 133, False, self.is_using_unicode)
                                for i in range(8)
                            ],
                        )
                        players[i].tile_units.append(p1_hands)
                        p1_chi1 = TileUnit(
                            TileUnitType.CHI,
                            FromWho.LEFT,
                            [
                                Tile(48, True, self.is_using_unicode),
                                Tile(52, True, self.is_using_unicode),
                                Tile(56, True, self.is_using_unicode),
                            ],
                        )
                        players[i].tile_units.append(p1_chi1)
                        p1_chi2 = TileUnit(
                            TileUnitType.CHI,
                            FromWho.LEFT,
                            [
                                Tile(60, True, self.is_using_unicode),
                                Tile(64, True, self.is_using_unicode),
                                Tile(68, True, self.is_using_unicode),
                            ],
                        )
                        players[i].tile_units.append(p1_chi2)

                for p in [players[0], players[1], players[2], players[3]]:
                    p.tile_units.append(
                        TileUnit(
                            TileUnitType.DISCARD,
                            FromWho.NONE,
                            [
                                Tile(124, True, self.is_using_unicode),
                                Tile(40, True, self.is_using_unicode, True),
                                Tile(128, True, self.is_using_unicode),
                                Tile(108, True, self.is_using_unicode, True),
                                Tile(112, True, self.is_using_unicode),
                                Tile(116, True, self.is_using_unicode),
                                Tile(120, True, self.is_using_unicode, True),
                                Tile(36, True, self.is_using_unicode),
                            ],
                        )
                    )

                table = MahjongTable(players[0], players[1], players[2], players[3])
                table.round = gamedata.public_observation.init_score.round + 1
                table.honba = gamedata.public_observation.init_score.honba
                table.riichi = 1
                table.last_player = 3
                table.last_action = 1
                table.wall_num = 36
                table.my_idx = gamedata.who

                if not table.check_num_tiles():
                    exit(1)

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
                            + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                            + (
                                ""
                                if tile.is_tsumogiri
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
                    if player_idx % 2 == 0:
                        if tile_unit.tile_unit_type == TileUnitType.HAND:
                            tiles += "".join(
                                [
                                    tile.char
                                    + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                                    for tile in tile_unit.tiles
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
                    else:
                        tiles += (
                            "\n"
                            + "\n".join([tile.char for tile in tile_unit.tiles])
                            + "\n"
                            + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                            + "\n"
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
                            + ("" if tile.char == "\U0001F004\uFE0E" else " ")
                            + (
                                ""
                                if tile.is_tsumogiri
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
                        str(p.score)
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
            get_wind_char(table.last_player, self.language)
            + ["'s turn now.\n", "の番です\n"][self.language]
        )
        system_info.append("ActionType:" + str(table.last_action))
        system_info = "".join(system_info)

        return "".join([board_info, players_info, system_info])

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

            score = Text(str(p.score), justify="center", style="yellow")

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
            hand_area = hand + "      " + chi + pon + open_kan + closed_kan + added_kan
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


def main():
    """
    >>> game_board = GameBoard("observations.json", "Observation", False, False, 0 , True)
    >>> print(game_board.show_by_text(game_board.load_data()))  # doctest: +NORMALIZE_WHITESPACE
    round:1 riichi:1 wall:36
    <BLANKLINE>
    EAST [ 25000 ] ワイルド
    <BLANKLINE>
    nw wd gd rd m1 m2 m3 m4       p4p5p6L p7p8p9L
    <BLANKLINE>
    wd  p2* gd  ew* sw  ww
    nw* p1
    <BLANKLINE>
    <BLANKLINE>
    SOUTH [ 25000 ] ミラクルおじさん
    <BLANKLINE>
    nw wd gd rd m1 m2 m3 m4       p4p5p6L p7p8p9L
    <BLANKLINE>
    wd  p2* gd  ew* sw  ww
    nw* p1
    <BLANKLINE>
    <BLANKLINE>
    WEST [ 25000 ] ASAPIN
    <BLANKLINE>
    nw wd gd rd m1 m2 m3 m4       p4p5p6L p7p8p9L
    <BLANKLINE>
    wd  p2* gd  ew* sw  ww
    nw* p1
    <BLANKLINE>
    <BLANKLINE>
    NORTH [ 25000 ] ＼(＾o＾)／★
    <BLANKLINE>
    nw wd gd rd m1 m2 m3 m4       p4p5p6L p7p8p9L
    <BLANKLINE>
    wd  p2* gd  ew* sw  ww
    nw* p1
    <BLANKLINE>
    <BLANKLINE>
    NORTH's turn now.
    ActionType:1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="observations.json")
    parser.add_argument("--mode", default="Observation")
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
    command = input()

    while command != "q":
        if command == "z":
            i = (i + turns - 1) % turns
        if command == "x":
            i = (i + 1) % turns

        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

        if args.rich:
            game_board.show_by_rich(game_data[i])
        else:
            print(game_board.show_by_text(game_data[i]))
        command = input()


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    main()
