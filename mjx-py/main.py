import argparse
from typing import List

from converter import get_actiontype, get_modifier, get_tile_char, get_wind_char

is_using_unicode = False
is_using_rich = False
language = 0


class Tile:
    def __init__(self, tile_id: int, is_open: bool = False):
        self.id = tile_id
        if not is_open:
            self.char = "\U0001F02B" if is_using_unicode else "#"
        else:
            self.char = get_tile_char(tile_id, is_using_unicode)


HandArea = List[Tile]


class Player:
    def __init__(
        self,
        player_idx: int,
        wind: int,
        init_score: int,
        init_hand: HandArea,
        name: str,
    ):
        self.player_idx = player_idx
        self.wind = wind
        self.wind_char = get_wind_char(wind + 4 * language)
        self.score = init_score
        self.hands_area = init_hand
        self.discard_area = []
        self.opens_area = []
        self.drawcount = 0
        self.is_riichi = False
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


class GameBoard:
    """
    GameBoardクラスは内部にMahjongTableクラスのオブジェクトを持ち、
    EventHistoryからの現在の状態の読み取りや、その表示などを行います。
    """

    def __init__(
        self, path: str, mode: str, show_players_name: bool = True,
    ):
        self.path = path
        self.name = mode
        self.show_players_name = show_players_name

    def rollout(self) -> MahjongTable:
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
            1, 0, 25000, [[0, [Tile(i * 4, True) for i in range(8)]]], "太郎"
        )
        player2 = Player(
            2, 1, 25000, [[0, [Tile(i * 4, False) for i in range(8)]]], "次郎"
        )
        player3 = Player(
            3, 2, 25000, [[0, [Tile(i * 4, False) for i in range(8)]]], "三郎"
        )
        player4 = Player(
            0, 3, 25000, [[0, [Tile(i * 4, False) for i in range(8)]]], "四郎"
        )

        player3.is_riichi = True

        player1.opens_area = [
            [0, [Tile(48, True), Tile(52, True), Tile(56, True)]],
            [0, [Tile(60, True), Tile(64, True), Tile(68, True)]],
        ]
        player2.opens_area = [
            [1, [Tile(72, True), Tile(73, True), Tile(74, True)]],
            [2, [Tile(76, True), Tile(77, True), Tile(78, True)]],
        ]
        player3.opens_area = [
            [3, [Tile(80, True), Tile(81, True), Tile(82, True), Tile(83, True)]],
            [4, [Tile(84, True), Tile(85, True), Tile(86, True), Tile(87, True)]],
        ]
        player4.opens_area = [
            [6, [Tile(96, True), Tile(97, True), Tile(98, True), Tile(99, True)]],
            [7, [Tile(100, True), Tile(101, True), Tile(102, True), Tile(103, True)]],
        ]

        for p in [player1, player2, player3, player4]:
            p.discard_area = [
                [8, [Tile(104, True)]],
                [-1, [Tile(108, True)]],
                [-1, [Tile(112, True)]],
                [8, [Tile(116, True)]],
                [8, [Tile(120, True)]],
                [-1, [Tile(124, True)]],
                [8, [Tile(128, True)]],
                [-1, [Tile(132, True)]],
            ]

        table = MahjongTable(player1, player2, player3, player4)
        table.round = 6  # 南2局
        table.honba = 1
        table.riichi = 1
        table.last_player = 3
        table.last_action = 1

        return table

    def is_num_of_tiles_ok(self, table: MahjongTable) -> bool:
        for p in table.players:
            for hands_info in p.hands_area:
                num_of_tiles = len(hands_info[1])
            num_of_tiles += len(p.opens_area) * 3

            if num_of_tiles < 13 or 14 < num_of_tiles:
                print("ERROR: The number of tiles is inaccurate.")
                print("player:", p.player_idx, num_of_tiles)
                return False
        return True

    def add_status(self, _tiles: list, is_opens=False) -> str:
        result = []
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
        if not self.is_num_of_tiles_ok(table):
            exit(1)

        board_info = []
        board_info.append(
            [
                f"round:{table.round}",
                get_wind_char((table.round - 1) // 4 + 4)
                + str((table.round - 1) % 4 + 1)
                + "局",
            ][language]
        )
        if table.honba > 0:
            board_info.append(
                " " + ["honba:" + str(table.honba), str(table.honba) + "本場"][language]
            )
        if table.riichi > 0:
            board_info.append(" " + ["riichi:", "供託"][language] + str(table.riichi))
        board_info.append("\n\n")
        board_info = "".join(board_info)

        players_info = []
        table.players.sort(key=lambda x: x.player_idx)
        for p in table.players:
            player_info = []
            player_info.append(
                get_wind_char(p.wind + 4 * language)
                + " [ "
                + "".join(
                    [
                        str(p.score)
                        + ([", riichi", ", リーチ"][language] if p.is_riichi else "")
                    ]
                )
                + " ]",
            )
            if self.show_players_name:
                player_info.append(" PLAYER NAME: " + p.name)
            player_info.append("\n\n")

            player_info.append(
                self.add_status(p.hands_area)
                + ", "
                + self.add_status(p.opens_area, True)
            )
            player_info.append("\n\n")

            discards = self.split_discards(p.discard_area)
            player_info.append(
                "\n".join([self.add_status(tiles) for tiles in discards])
            )
            player_info.append("\n\n\n")
            players_info.append("".join(player_info))
        players_info = "".join(players_info)

        system_info = []
        system_info.append(
            get_wind_char(table.last_player + 4 * language)
            + ["'s turn now.\n", "の番です\n"][language]
        )
        system_info.append("ActionType:" + get_actiontype(table.last_action))
        system_info = "".join(system_info)

        return "".join([board_info, players_info, system_info])


def main():
    # 将来的に引数として設定すべきなのは、
    # ファイルpath, State/Observation, rich(w or w/o), unicode(w or w/o), 名前表示, 言語

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="2010091009gm-00a9-0000-83af2648&tw=2.json")
    parser.add_argument("--mode", default="Obs")
    parser.add_argument("--uni", action="store_true")
    parser.add_argument("--rich", action="store_true")
    parser.add_argument("--show_name", action="store_true")
    parser.add_argument("--lang", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    global is_using_unicode
    global is_using_rich
    global language
    is_using_unicode = args.uni
    is_using_rich = args.rich
    language = args.lang

    game_board = GameBoard(args.path, args.mode, args.show_name)
    print(game_board.show(game_board.rollout()))


if __name__ == "__main__":
    main()
