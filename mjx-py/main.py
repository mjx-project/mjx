import argparse
from typing import List

from converter import (
    get_actiontype,
    get_fromwho,
    get_tile_char,
    get_tile_unicode,
    get_wind_char,
)


class Tile:
    def __init__(self, tile_id: int, is_open: bool = False):
        self.id = tile_id
        self.is_open = is_open
        if not is_open:
            self.char = "#"
            self.unicode = "\U0001F02B"
        else:
            self.char = get_tile_char(tile_id)
            self.unicode = get_tile_unicode(tile_id)


PlayArea = List[Tile]


class Player:
    def __init__(
        self, player_idx: int, wind: int, init_score: int, init_hand: PlayArea
    ):
        self.player_idx = player_idx
        self.wind = wind
        self.wind_char = get_wind_char(wind)
        self.score = init_score
        self.hands_area = init_hand
        self.discard_area = []
        self.chi_area = []
        self.pon_area = []
        self.kan_closed_area = []
        self.kan_opened_area = []
        self.kan_added_area = []
        self.drawcount = 0


class MahjongTable:
    """
    MahjongTableクラスは場の情報（プレイヤーの手牌や河など）を保持します。
    """

    def __init__(
        self, player1: Player, player2: Player, player3: Player, player4: Player,
    ):
        self.players = [player1, player2, player3, player4]
        self.last_action = 0  # 0-10
        self.last_player = 0  # 0-3


class GameBoard:
    """
    GameBoardクラスは内部にMahjongTableクラスのオブジェクトを持ち、
    EventHistoryからの現在の状態の読み取りや、その表示などを行います。
    """

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

    def rollout(self) -> None:
        """
        ここでEventHistoryから現在の状態を読み取る予定です。

        with open(self.path, "r", errors="ignore") as f:
            for line in f:
                gamedata = mjxproto.Observation()
                gamedata.from_json(line)

                # 入力作業
        
        # 今はテスト用に、直に打ち込んでいます。
        """
        player1 = Player(0, 0, 25000, [Tile(i * 4, True) for i in range(14)])
        player2 = Player(1, 1, 25000, [Tile(i * 4, False) for i in range(14)])
        player3 = Player(2, 2, 25000, [Tile(i * 4, False) for i in range(14)])
        player4 = Player(3, 3, 25000, [Tile(i * 4, False) for i in range(14)])

        for p in [player1, player2, player3, player4]:

            p.chi_area = [
                [Tile(48, True), Tile(52, True), Tile(56, True)],
                [Tile(60, True), Tile(64, True), Tile(68, True)],
            ]
            p.pon_area = [
                [0, [Tile(72, True), Tile(73, True), Tile(74, True)]],
                [1, [Tile(76, True), Tile(77, True), Tile(78, True)]],
            ]
            p.kan_closed_area = [
                [Tile(80, False), Tile(81, True), Tile(82, True), Tile(83, False)],
                [Tile(84, False), Tile(85, True), Tile(86, True), Tile(87, False)],
            ]
            p.kan_opened_area = [
                [2, [Tile(88, True), Tile(89, True), Tile(90, True), Tile(91, True)]],
                [0, [Tile(92, True), Tile(93, True), Tile(94, True), Tile(95, True)]],
            ]
            p.kan_added_area = [
                [1, [Tile(96, True), Tile(97, True), Tile(98, True), Tile(99, True)]],
                [
                    2,
                    [
                        Tile(100, True),
                        Tile(101, True),
                        Tile(102, True),
                        Tile(103, True),
                    ],
                ],
            ]
            p.discard_area = [
                Tile(104, True),
                Tile(108, True),
                Tile(112, True),
                Tile(116, True),
            ]

        table = MahjongTable(player1, player2, player3, player4)
        table.last_player = 3
        table.last_action = 1

        self.table = table

    def show_all(self) -> None:
        for p in self.table.players:
            print(get_wind_char(p.wind))
            print("SCORE:", p.score)
            print("手牌: ", [tile.char for tile in p.hands_area])
            print(
                "チー: ",
                ["".join([tile.char for tile in tiles]) for tiles in p.chi_area],
            )
            print(
                "ポン: ",
                [
                    "".join([tile.char for tile in tiles[1]]) + get_fromwho(tiles[0])
                    for tiles in p.pon_area
                ],
            )
            print(
                "暗槓: ",
                ["".join([tile.char for tile in tiles]) for tiles in p.kan_closed_area],
            )
            print(
                "明槓: ",
                [
                    "".join([tile.char for tile in tiles[1]]) + get_fromwho(tiles[0])
                    for tiles in p.kan_opened_area
                ],
            )
            print(
                "加槓: ",
                [
                    "".join([tile.char for tile in tiles[1]]) + get_fromwho(tiles[0])
                    for tiles in p.kan_added_area
                ],
            )
            print(
                "河  : ", [tile.char for tile in p.discard_area],
            )
            print()
        print(get_wind_char(self.table.last_player), "の番です")
        print("ActionType:", get_actiontype(self.table.last_action))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="2010091009gm-00a9-0000-83af2648&tw=2.json")
    args = parser.parse_args()

    game_board = GameBoard(args.path, "Observation")
    game_board.rollout()
    game_board.show_all()


if __name__ == "__main__":
    main()
