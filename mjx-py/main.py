import argparse
from converter import TileUnitType, FromWho
from converter import get_modifier, get_tile_char, get_wind_char
from rich import print
from rich.layout import Layout
from rich.panel import Panel
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
        if not is_open:
            self.char = "\U0001F02B" if is_using_unicode else "#"
        else:
            self.char = get_tile_char(tile_id, is_using_unicode)
        if is_tsumogiri:
            self.char += " *"
        if tile_id != 33:
            self.char += ""


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
    def __init__(
        self,
        player_idx: int,
        wind: int,
        score: int,
        name: str,
    ):
        self.player_idx = player_idx
        self.wind = wind
        self.score = score
        self.tile_units = []
        self.is_declared_riichi = False
        self.name = name

    def add_status():
        pass


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

    def check_num_tiles(self) -> bool:
        for p in self.players:
            for tile_unit in p.tile_units:
                if tile_unit.tile_unit_type == TileUnitType.HAND:
                    num_of_tiles = len(tile_unit.tiles)
                elif tile_unit.tile_unit_type != TileUnitType.DISCARD:
                    num_of_tiles += 3

            if num_of_tiles < 13 or 14 < num_of_tiles:
                print("ERROR: The number of tiles is inaccurate.")
                print("player:", p.player_idx, num_of_tiles)
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
        self.name = mode
        self.is_using_unicode = is_using_unicode
        self.is_using_rich = is_using_rich
        self.language = language
        self.show_name = show_name
        self.my_idx = 0

        self.layout = Layout()
        self.layout.split_column(
            Layout(name="info"),
            Layout(" ", name="space"),
            Layout(name="table"),
        )
        self.layout["info"].size = 3
        self.layout["space"].size = 3
        # self.layout["table"].size = 60

        self.layout["table"].split_column(
            Layout(name="upper1"),
            Layout(name="middle1"),
            Layout(name="lower1"),
        )

        self.layout["upper1"].size = 3
        self.layout["middle1"].size = 25
        self.layout["lower1"].size = 3

        self.layout["upper1"].split_row(
            Layout(" "),
            Layout(name="hand3"),
            Layout(" "),
        )
        self.layout["hand3"].ratio = 6

        self.layout["middle1"].split_row(
            Layout(name="hand4"),
            Layout(name="middle2"),
            Layout(name="hand2"),
        )
        self.layout["middle2"].ratio = 6

        self.layout["lower1"].split_row(
            Layout(" "),
            Layout(name="hand1"),
            Layout(" "),
        )
        self.layout["hand1"].ratio = 6

        self.layout["middle2"].split_column(
            Layout(name="upper2"),
            Layout(name="middle3"),
            Layout(name="lower2"),
        )

        self.layout["upper2"].split_row(
            Layout(" "),
            Layout(name="discard3"),
            Layout(" "),
        )
        # self.layout["discard3"].ratio = 2

        self.layout["middle3"].split_row(
            Layout(name="discard4"),
            Layout(name="center"),
            Layout(name="discard2"),
        )
        # self.layout["center"].ratio = 2

        self.layout["lower2"].split_row(
            Layout(" "),
            Layout(name="discard1"),
            Layout(" "),
        )
        # self.layout["discard1"].ratio = 2
        self.layout["center"].update(Panel(" "))

    def load_data(self) -> MahjongTable:
        """
        ここでEventHistoryから現在の状態を読み取る予定です。

        with open(self.path, "r", errors="ignore") as f:
            for line in f:
                gamedata = mjxproto.Observation()
                gamedata.from_json(line)

                # 入力作業

        今はテスト用に、直に打ち込んでいます。
        """
        player1 = Player(
            0,
            0,
            25000,
            "太郎",
        )
        player2 = Player(
            1,
            1,
            25000,
            "次郎",
        )
        player3 = Player(
            2,
            2,
            25000,
            "三郎",
        )
        player4 = Player(
            3,
            3,
            25000,
            "四郎",
        )

        # player1
        p1_hands = TileUnit(
            TileUnitType.HAND,
            FromWho.NONE,
            [Tile(i * 4, True, self.is_using_unicode) for i in range(8)],
        )
        player1.tile_units.append(p1_hands)
        p1_chi1 = TileUnit(
            TileUnitType.CHI,
            FromWho.LEFT,
            [
                Tile(48, True, self.is_using_unicode),
                Tile(52, True, self.is_using_unicode),
                Tile(56, True, self.is_using_unicode),
            ],
        )
        player1.tile_units.append(p1_chi1)
        p1_chi2 = TileUnit(
            TileUnitType.CHI,
            FromWho.LEFT,
            [
                Tile(60, True, self.is_using_unicode),
                Tile(64, True, self.is_using_unicode),
                Tile(68, True, self.is_using_unicode),
            ],
        )
        player1.tile_units.append(p1_chi2)

        # player2
        p2_hands = TileUnit(
            TileUnitType.HAND,
            FromWho.NONE,
            [Tile(i * 4, False, self.is_using_unicode) for i in range(8)],
        )
        player2.tile_units.append(p2_hands)
        p2_pon1 = TileUnit(
            TileUnitType.PON,
            FromWho.MID,
            [
                Tile(72, True, self.is_using_unicode),
                Tile(73, True, self.is_using_unicode),
                Tile(74, True, self.is_using_unicode),
            ],
        )
        player2.tile_units.append(p2_pon1)
        p2_pon2 = TileUnit(
            TileUnitType.PON,
            FromWho.RIGHT,
            [
                Tile(76, True, self.is_using_unicode),
                Tile(77, True, self.is_using_unicode),
                Tile(78, True, self.is_using_unicode),
            ],
        )
        player2.tile_units.append(p2_pon2)

        # player3
        p3_hands = TileUnit(
            TileUnitType.HAND,
            FromWho.NONE,
            [Tile(i * 4, False, self.is_using_unicode) for i in range(8)],
        )
        player3.tile_units.append(p3_hands)
        p3_kan1 = TileUnit(
            TileUnitType.OPEN_KAN,
            FromWho.RIGHT,
            [
                Tile(80, True, self.is_using_unicode),
                Tile(81, True, self.is_using_unicode),
                Tile(82, True, self.is_using_unicode),
                Tile(83, True, self.is_using_unicode),
            ],
        )
        player3.tile_units.append(p3_kan1)
        p3_kan2 = TileUnit(
            TileUnitType.OPEN_KAN,
            FromWho.RIGHT,
            [
                Tile(84, True, self.is_using_unicode),
                Tile(85, True, self.is_using_unicode),
                Tile(86, True, self.is_using_unicode),
                Tile(87, True, self.is_using_unicode),
            ],
        )
        player3.tile_units.append(p3_kan2)
        player3.is_declared_riichi = True

        # player4
        p4_hands = TileUnit(
            TileUnitType.HAND,
            FromWho.NONE,
            [Tile(i * 4, False, self.is_using_unicode) for i in range(8)],
        )
        player4.tile_units.append(p4_hands)
        p4_kan1 = TileUnit(
            TileUnitType.CLOSED_KAN,
            FromWho.RIGHT,
            [
                Tile(96, True, self.is_using_unicode),
                Tile(97, True, self.is_using_unicode),
                Tile(98, True, self.is_using_unicode),
                Tile(99, True, self.is_using_unicode),
            ],
        )
        player4.tile_units.append(p4_kan1)
        p4_kan2 = TileUnit(
            TileUnitType.ADDED_KAN,
            FromWho.LEFT,
            [
                Tile(100, True, self.is_using_unicode),
                Tile(101, True, self.is_using_unicode),
                Tile(102, True, self.is_using_unicode),
                Tile(103, True, self.is_using_unicode),
            ],
        )
        player4.tile_units.append(p4_kan2)

        for p in [player1, player2, player3, player4]:
            p.tile_units.append(
                TileUnit(
                    TileUnitType.DISCARD,
                    FromWho.NONE,
                    [
                        Tile(104, True, self.is_using_unicode),
                        Tile(108, True, self.is_using_unicode, True),
                        Tile(112, True, self.is_using_unicode),
                        Tile(116, True, self.is_using_unicode),
                        Tile(120, True, self.is_using_unicode, True),
                        Tile(124, True, self.is_using_unicode),
                        Tile(128, True, self.is_using_unicode, True),
                        Tile(132, True, self.is_using_unicode),
                    ],
                )
            )

        table = MahjongTable(player1, player2, player3, player4)
        table.round = 6  # 南2局
        table.honba = 1
        table.riichi = 1
        table.last_player = 3
        table.last_action = 1

        if not table.check_num_tiles():
            exit(1)

        self.table = table
        return table

    def get_modified_tiles(self, player_idx: int, tile_unit_type: TileUnitType):
        if self.is_using_rich:
            tiles = ""
            for tile_unit in self.table.players[player_idx].tile_units:
                if tile_unit.tile_unit_type == tile_unit_type:
                    if tile_unit.tile_unit_type == TileUnitType.DISCARD:
                        discards = [tile.char for tile in tile_unit.tiles]
                        tiles += "\n".join(
                            [
                                " ".join(discards[idx : idx + 6])
                                for idx in range(0, len(discards), 6)
                            ]
                        )
                        break
                    if player_idx % 2 == 0:
                        if tile_unit.tile_unit_type == TileUnitType.HAND:
                            tiles += " ".join([tile.char for tile in tile_unit.tiles])
                            break
                        tiles += (
                            (" " if self.is_using_unicode else "").join(
                                [tile.char for tile in tile_unit.tiles]
                            )
                            + (" " if self.is_using_unicode else "")
                            + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                        )
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
            for tile_unit in self.table.players[player_idx].tile_units:
                if tile_unit.tile_unit_type == tile_unit_type:
                    if tile_unit.tile_unit_type == TileUnitType.HAND:
                        tiles += " ".join([tile.char for tile in tile_unit.tiles])
                        break
                    if tile_unit.tile_unit_type == TileUnitType.DISCARD:
                        discards = [tile.char for tile in tile_unit.tiles]
                        tiles += " ".join(
                            [
                                " ".join(discards[idx : idx + 6])
                                for idx in range(0, len(discards), 6)
                            ]
                        )
                        break

                    tiles += (
                        (" " if self.is_using_unicode else "").join(
                            [tile.char for tile in tile_unit.tiles]
                        )
                        + (" " if self.is_using_unicode else "")
                        + get_modifier(tile_unit.from_who, tile_unit.tile_unit_type)
                    )

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
        board_info.append("\n\n")
        board_info = "".join(board_info)
        return board_info

    def show_by_text(self, table: MahjongTable) -> str:
        self.my_idx = table.my_idx
        board_info = self.get_board_info(table)

        players_info = []
        table.players.sort(key=lambda x: x.player_idx)
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

            hand = self.get_modified_tiles(i, TileUnitType.HAND)
            chi = self.get_modified_tiles(i, TileUnitType.CHI)
            pon = self.get_modified_tiles(i, TileUnitType.PON)
            open_kan = self.get_modified_tiles(i, TileUnitType.OPEN_KAN)
            closed_kan = self.get_modified_tiles(i, TileUnitType.CLOSED_KAN)
            added_kan = self.get_modified_tiles(i, TileUnitType.ADDED_KAN)
            hand_area = hand + "  " + chi + pon + open_kan + closed_kan + added_kan
            player_info.append(hand_area)
            player_info.append("\n\n")

            discards = self.get_modified_tiles(i, TileUnitType.DISCARD)
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
            Panel(Text(self.get_board_info(table), justify="center"))
        )

        table.players.sort(key=lambda x: (x.player_idx - self.my_idx) % 4)
        hands_idx = ["hand1", "hand2", "hand3", "hand4"]
        discards_idx = ["discard1", "discard2", "discard3", "discard4"]

        for i, p in enumerate(table.players):
            hand = self.get_modified_tiles(i, TileUnitType.HAND)
            chi = self.get_modified_tiles(i, TileUnitType.CHI)
            pon = self.get_modified_tiles(i, TileUnitType.PON)
            open_kan = self.get_modified_tiles(i, TileUnitType.OPEN_KAN)
            closed_kan = self.get_modified_tiles(i, TileUnitType.CLOSED_KAN)
            added_kan = self.get_modified_tiles(i, TileUnitType.ADDED_KAN)
            hand_area = hand + "  " + chi + pon + open_kan + closed_kan + added_kan
            self.layout[hands_idx[i]].update(
                Panel(Text(hand_area, justify="center", no_wrap=True))
            )
            discards = self.get_modified_tiles(i, TileUnitType.DISCARD)
            self.layout[discards_idx[i]].update(
                Panel(Text(discards, justify="left", no_wrap=True))
            )

        print(self.layout)


def main():
    # 将来的に引数として設定すべきなのは、
    # ファイルpath, State/Observation, rich(w or w/o), unicode(w or w/o), 名前表示, 言語

    """
    >>> game_board = GameBoard("hogepath", "Observation", False, False, 0 , True)
    >>> print(game_board.show_by_text(game_board.load_data()))  # doctest: +NORMALIZE_WHITESPACE
    round:6 honba:1 riichi:1
    <BLANKLINE>
    EAST [ 25000 ] 太郎
    <BLANKLINE>
    m1 m2 m3 m4 m5 m6 m7 m8  p4p5p6L p7p8p9L
    <BLANKLINE>
    s9 ew * sw ww nw * wd gd * rd
    <BLANKLINE>
    <BLANKLINE>
    SOUTH [ 25000 ] 次郎
    <BLANKLINE>
    # # # # # # # #  s1s1s1M s2s2s2R
    <BLANKLINE>
    s9 ew * sw ww nw * wd gd * rd
    <BLANKLINE>
    <BLANKLINE>
    WEST [ 25000, riichi ] 三郎
    <BLANKLINE>
    # # # # # # # #  s3s3s3s3R s4s4s4s4R
    <BLANKLINE>
    s9 ew * sw ww nw * wd gd * rd
    <BLANKLINE>
    <BLANKLINE>
    NORTH [ 25000 ] 四郎
    <BLANKLINE>
    # # # # # # # #  s7s7s7s7R s8s8s8s8L(Add)
    <BLANKLINE>
    s9 ew * sw ww nw * wd gd * rd
    <BLANKLINE>
    <BLANKLINE>
    NORTH's turn now.
    ActionType:1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="2010091009gm-00a9-0000-83af2648&tw=2.json")
    parser.add_argument("--mode", default="Obs")
    parser.add_argument("--uni", action="store_true")
    parser.add_argument("--rich", action="store_true")
    parser.add_argument("--show_name", action="store_true")
    parser.add_argument("--lang", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    game_board = GameBoard(
        args.path, args.mode, args.uni, args.rich, args.lang, args.show_name
    )
    if args.rich:
        game_board.show_by_rich(game_board.load_data())
    else:
        print(game_board.show_by_text(game_board.load_data()))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    main()
