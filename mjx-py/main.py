import argparse

from converter import get_modifier, get_tile_char, get_wind_char


class Tile:
    def __init__(
        self, tile_id: int, is_open: bool = False, is_using_unicode: bool = False
    ):
        self.id = tile_id
        if not is_open:
            self.char = "\U0001F02B" if is_using_unicode else "#"
        else:
            self.char = get_tile_char(tile_id, is_using_unicode)


class Player:
    def __init__(
        self, player_idx: int, wind: int, score: int, hands: list, name: str,
    ):
        self.player_idx = player_idx
        self.wind = wind
        self.score = score
        self.hands = hands
        self.discard = []
        self.opens = []
        self.is_declared_riichi = False
        self.name = name


class MahjongTable:
    """
    MahjongTableクラスは場の情報（プレイヤーの手牌や河など）を保持します。
    """

    def __init__(
        self, player1: Player, player2: Player, player3: Player, player4: Player,
    ):
        self.players = [player1, player2, player3, player4]
        self.riichi = 0
        self.round = 0
        self.honba = 0
        self.last_action = 0  # 0-10
        self.last_player = 0  # 0-3

    def check_num_tiles(self) -> bool:
        for p in self.players:
            for hands_info in p.hands:
                num_of_tiles = len(hands_info[1])
            num_of_tiles += len(p.opens) * 3

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

        """
        [<修飾文字番号>,[<Tile>,<Tile>,...]]
        """
        player1 = Player(
            1,
            0,
            25000,
            [[0, [Tile(i * 4, True, self.is_using_unicode) for i in range(8)]]],
            "太郎",
        )
        player2 = Player(
            2,
            1,
            25000,
            [[0, [Tile(i * 4, False, self.is_using_unicode) for i in range(8)]]],
            "次郎",
        )
        player3 = Player(
            3,
            2,
            25000,
            [[0, [Tile(i * 4, False, self.is_using_unicode) for i in range(8)]]],
            "三郎",
        )
        player4 = Player(
            0,
            3,
            25000,
            [[0, [Tile(i * 4, False, self.is_using_unicode) for i in range(8)]]],
            "四郎",
        )

        player3.is_declared_riichi = True

        player1.opens = [
            [
                0,
                [
                    Tile(48, True, self.is_using_unicode),
                    Tile(52, True, self.is_using_unicode),
                    Tile(56, True, self.is_using_unicode),
                ],
            ],
            [
                0,
                [
                    Tile(60, True, self.is_using_unicode),
                    Tile(64, True, self.is_using_unicode),
                    Tile(68, True, self.is_using_unicode),
                ],
            ],
        ]
        player2.opens = [
            [
                1,
                [
                    Tile(72, True, self.is_using_unicode),
                    Tile(73, True, self.is_using_unicode),
                    Tile(74, True, self.is_using_unicode),
                ],
            ],
            [
                2,
                [
                    Tile(76, True, self.is_using_unicode),
                    Tile(77, True, self.is_using_unicode),
                    Tile(78, True, self.is_using_unicode),
                ],
            ],
        ]
        player3.opens = [
            [
                3,
                [
                    Tile(80, True, self.is_using_unicode),
                    Tile(81, True, self.is_using_unicode),
                    Tile(82, True, self.is_using_unicode),
                    Tile(83, True, self.is_using_unicode),
                ],
            ],
            [
                4,
                [
                    Tile(84, True, self.is_using_unicode),
                    Tile(85, True, self.is_using_unicode),
                    Tile(86, True, self.is_using_unicode),
                    Tile(87, True, self.is_using_unicode),
                ],
            ],
        ]
        player4.opens = [
            [
                6,
                [
                    Tile(96, True, self.is_using_unicode),
                    Tile(97, True, self.is_using_unicode),
                    Tile(98, True, self.is_using_unicode),
                    Tile(99, True, self.is_using_unicode),
                ],
            ],
            [
                7,
                [
                    Tile(100, True, self.is_using_unicode),
                    Tile(101, True, self.is_using_unicode),
                    Tile(102, True, self.is_using_unicode),
                    Tile(103, True, self.is_using_unicode),
                ],
            ],
        ]

        for p in [player1, player2, player3, player4]:
            p.discard = [
                [8, [Tile(104, True, self.is_using_unicode)]],
                [-1, [Tile(108, True, self.is_using_unicode)]],
                [-1, [Tile(112, True, self.is_using_unicode)]],
                [8, [Tile(116, True, self.is_using_unicode)]],
                [8, [Tile(120, True, self.is_using_unicode)]],
                [-1, [Tile(124, True, self.is_using_unicode)]],
                [8, [Tile(128, True, self.is_using_unicode)]],
                [-1, [Tile(132, True, self.is_using_unicode)]],
            ]

        table = MahjongTable(player1, player2, player3, player4)
        table.round = 6  # 南2局
        table.honba = 1
        table.riichi = 1
        table.last_player = 3
        table.last_action = 1

        if not table.check_num_tiles():
            exit(1)

        return table

    def add_status(self, _tiles: list, is_opens=False) -> str:
        result = []
        if self.is_using_unicode:
            for x, tiles in _tiles:
                if is_opens:
                    result.append(
                        " ".join([tile.char for tile in tiles]) + " " + get_modifier(x)
                    )
                else:
                    result.append(
                        " ".join([tile.char for tile in tiles]) + " " + get_modifier(x)
                    )
            if is_opens:
                return ", ".join(result)
            return " ".join(result)

        for x, tiles in _tiles:
            if is_opens:
                result.append("".join([tile.char for tile in tiles]) + get_modifier(x))
            else:
                result.append(" ".join([tile.char for tile in tiles]) + get_modifier(x))
        if is_opens:
            return ", ".join(result)
        return " ".join(result)

    def split_discards(self, discards: list) -> list:
        return [discards[idx : idx + 6] for idx in range(0, len(discards), 6)]

    def show(self, table: MahjongTable) -> str:
        if not table.check_num_tiles():
            exit(1)

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

        players_info = []
        table.players.sort(key=lambda x: x.player_idx)
        for p in table.players:
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

            player_info.append(
                self.add_status(p.hands) + ", " + self.add_status(p.opens, True)
            )
            player_info.append("\n\n")

            discards = self.split_discards(p.discard)
            player_info.append(
                "\n".join([self.add_status(tiles) for tiles in discards])
            )
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


def main():
    # 将来的に引数として設定すべきなのは、
    # ファイルpath, State/Observation, rich(w or w/o), unicode(w or w/o), 名前表示, 言語

    """
    >>> game_board = GameBoard("hogepath", "Observation", False, False, 0 , True)
    >>> print(game_board.show(game_board.load_data()))  # doctest: +NORMALIZE_WHITESPACE
    round:6 honba:1 riichi:1
    <BLANKLINE>
    NORTH [ 25000 ] 四郎
    <BLANKLINE>
    # # # # # # # #, s7s7s7s7_M(Add), s8s8s8s8_L(Add)
    <BLANKLINE>
    s9* ew  sw  ww* nw* wd
    gd* rd
    <BLANKLINE>
    <BLANKLINE>
    EAST [ 25000 ] 太郎
    <BLANKLINE>
    m1 m2 m3 m4 m5 m6 m7 m8, p4p5p6, p7p8p9
    <BLANKLINE>
    s9* ew  sw  ww* nw* wd
    gd* rd
    <BLANKLINE>
    <BLANKLINE>
    SOUTH [ 25000 ] 次郎
    <BLANKLINE>
    # # # # # # # #, s1s1s1_R, s2s2s2_M
    <BLANKLINE>
    s9* ew  sw  ww* nw* wd
    gd* rd
    <BLANKLINE>
    <BLANKLINE>
    WEST [ 25000, riichi ] 三郎
    <BLANKLINE>
    # # # # # # # #, s3s3s3s3_L, s4s4s4s4_S
    <BLANKLINE>
    s9* ew  sw  ww* nw* wd
    gd* rd
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
    print(game_board.show(game_board.load_data()))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # main()
