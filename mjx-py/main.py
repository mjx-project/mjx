import argparse
from typing import List

from rich.console import Console
from rich.table import Table

from converter import get_tile_char, get_tile_unicode, get_wind_char


class TileClass:
    def __init__(self, tile_id: int, is_open: bool = False, is_vertical: bool = False):
        self.id = tile_id
        self.is_open = is_open
        self.is_vertical = is_vertical
        if not is_open:
            self.char = "#"
            self.unicode = "#"
        elif is_vertical:
            self.char = "|" + get_tile_char(tile_id) + "|"
            self.unicode = "|" + get_tile_unicode(tile_id) + "|"
        else:
            self.char = get_tile_char(tile_id)
            self.unicode = get_tile_unicode(tile_id)


PlayArea = List[TileClass]


class PlayerClass:
    def __init__(self, id: int, init_hand: PlayArea):
        self.id = id
        self.wind = get_wind_char(id)
        self.hands_area = init_hand
        self.discard_area = []
        self.chi_area = []
        self.pon_area = []
        self.kan_closed_area = []
        self.kan_opened_area = []
        self.kan_added_area = []
        self.drawcount = 0


"""
MahjongTableClassには、各プレイヤーの手牌や河の情報が格納されています。
GameBoardClassは内部にMahjongTableClass型のオブジェクトを持ち、
状態の代入や画面への表示を行います。
"""


class MahjongTableClass:
    def __init__(
        self,
        player1: PlayerClass,
        player2: PlayerClass,
        player3: PlayerClass,
        player4: PlayerClass,
    ):
        self.players = [player1, player2, player3, player4]
        self.hands = []
        self.discs = []
        self.chis = []
        self.pons = []
        self.c_kans = []
        self.o_kans = []
        self.a_kans = []


class GameBoardClass:
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
        player1 = PlayerClass(0, [TileClass(i * 4, True) for i in range(12)])
        player2 = PlayerClass(1, [TileClass(i * 4, False) for i in range(12)])
        player3 = PlayerClass(2, [TileClass(i * 4, False) for i in range(12)])
        player4 = PlayerClass(3, [TileClass(i * 4, False) for i in range(12)])

        for p in [player1, player2, player3, player4]:

            p.chi_area = [
                [TileClass(48, True, True), TileClass(52, True), TileClass(56, True)],
                [TileClass(60, True), TileClass(64, True, True), TileClass(68, True)],
            ]
            p.pon_area = [
                [TileClass(72, True), TileClass(73, True), TileClass(74, True, True)],
                [TileClass(76, True, True), TileClass(77, True), TileClass(78, True)],
            ]
            p.kan_closed_area = [
                [
                    TileClass(80, False),
                    TileClass(81, True),
                    TileClass(82, True),
                    TileClass(83, False),
                ],
                [
                    TileClass(84, False),
                    TileClass(85, True),
                    TileClass(86, True),
                    TileClass(87, False),
                ],
            ]
            p.kan_opened_area = [
                [
                    TileClass(88, True, True),
                    TileClass(89, True),
                    TileClass(90, True),
                    TileClass(91, True),
                ],
                [
                    TileClass(92, True),
                    TileClass(93, True, True),
                    TileClass(94, True),
                    TileClass(95, True),
                ],
            ]
            p.kan_added_area = [
                [
                    TileClass(96, True, True),
                    TileClass(97, True, True),
                    TileClass(98, True),
                    TileClass(99, True),
                ],
                [
                    TileClass(100, True),
                    TileClass(101, True, True),
                    TileClass(102, True, True),
                    TileClass(103, True),
                ],
            ]
            p.discard_area = [
                TileClass(104, True),
                TileClass(108, True),
                TileClass(112, True),
                TileClass(116, True),
            ]

        table = MahjongTableClass(player1, player2, player3, player4)
        self.table = table

    def show_all(self) -> None:
        console = Console()

        table = Table(show_header=True, header_style="bold magenta")

        for p in self.table.players:
            self.table.hands.append([i.char for i in p.hands_area])
            self.table.discs.append([i.char for i in p.discard_area])

            # [[0,4,8],[12,16,20]] -> [[一二三],[四五六]]
            self.table.chis.append(["".join([j.char for j in i]) for i in p.chi_area])
            self.table.pons.append(["".join([j.char for j in i]) for i in p.pon_area])
            self.table.c_kans.append(
                ["".join([j.char for j in i]) for i in p.kan_closed_area]
            )
            self.table.o_kans.append(
                ["".join([j.char for j in i]) for i in p.kan_opened_area]
            )
            self.table.a_kans.append(
                ["".join([j.char for j in i]) for i in p.kan_added_area]
            )

        table.add_column("")
        table.add_column(self.table.players[0].wind)
        table.add_column(self.table.players[1].wind)
        table.add_column(self.table.players[2].wind)
        table.add_column(self.table.players[3].wind)

        table.add_row(
            "手牌",
            " ".join(self.table.hands[0]),
            " ".join(self.table.hands[1]),
            " ".join(self.table.hands[2]),
            " ".join(self.table.hands[3]),
        )
        table.add_row(
            "チー",
            " ".join(self.table.chis[0]),
            " ".join(self.table.chis[1]),
            " ".join(self.table.chis[2]),
            " ".join(self.table.chis[3]),
        )
        table.add_row(
            "ポン",
            " ".join(self.table.pons[0]),
            " ".join(self.table.pons[1]),
            " ".join(self.table.pons[2]),
            " ".join(self.table.pons[3]),
        )
        table.add_row(
            "暗槓",
            " ".join(self.table.c_kans[0]),
            " ".join(self.table.c_kans[1]),
            " ".join(self.table.c_kans[2]),
            " ".join(self.table.c_kans[3]),
        )
        table.add_row(
            "明槓",
            " ".join(self.table.o_kans[0]),
            " ".join(self.table.o_kans[1]),
            " ".join(self.table.o_kans[2]),
            " ".join(self.table.o_kans[3]),
        )
        table.add_row(
            "加槓",
            " ".join(self.table.a_kans[0]),
            " ".join(self.table.a_kans[1]),
            " ".join(self.table.a_kans[2]),
            " ".join(self.table.a_kans[3]),
        )
        table.add_row(
            "河",
            " ".join(self.table.discs[0]),
            " ".join(self.table.discs[1]),
            " ".join(self.table.discs[2]),
            " ".join(self.table.discs[3]),
        )

        console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="2010091009gm-00a9-0000-83af2648&tw=2.json")
    args = parser.parse_args()

    game_board = GameBoardClass(args.path, "Observation")
    game_board.rollout()
    game_board.show_all()


if __name__ == "__main__":
    main()
