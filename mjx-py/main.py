import argparse
from typing import List

from converter import get_actiontype, get_modifier, get_tile_char, get_wind_char


class Tile:
    def __init__(
        self, tile_id: int, is_open: bool = False, is_using_unicode: bool = False
    ):
        self.id = tile_id
        self.is_open = is_open
        if not is_open:
            self.char = "\U0001F02B" if is_using_unicode else "#"
        else:
            self.char = get_tile_char(tile_id, is_using_unicode)


HandArea = List[Tile]


class Player:
    def __init__(
        self, player_idx: int, wind: int, init_score: int, init_hand: HandArea
    ):
        self.player_idx = player_idx
        self.wind = wind
        self.wind_char = get_wind_char(wind)
        self.score = init_score
        self.hands_area = init_hand
        self.discard_area = []
        self.opens_area = []
        self.drawcount = 0
        self.is_riichi = False


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
        player1 = Player(1, 0, 25000, [[0, [Tile(i * 4, True) for i in range(8)]]])
        player2 = Player(2, 1, 25000, [[0, [Tile(i * 4, False) for i in range(8)]]])
        player3 = Player(3, 2, 25000, [[0, [Tile(i * 4, False) for i in range(8)]]])
        player4 = Player(0, 3, 25000, [[0, [Tile(i * 4, False) for i in range(8)]]])

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
            [4, [Tile(80, True), Tile(81, True), Tile(82, True), Tile(83, True)]],
            [4, [Tile(84, True), Tile(85, True), Tile(86, True), Tile(87, True)]],
        ]
        player4.opens_area = [
            [6, [Tile(96, True), Tile(97, True), Tile(98, True), Tile(99, True)]],
            [7, [Tile(100, True), Tile(101, True), Tile(102, True), Tile(103, True)]],
        ]

        for p in [player1, player2, player3, player4]:
            p.discard_area = [
                [8, [Tile(104, True)]],
                [0, [Tile(108, True)]],
                [8, [Tile(112, True)]],
                [0, [Tile(116, True)]],
                [8, [Tile(120, True)]],
                [0, [Tile(124, True)]],
                [8, [Tile(128, True)]],
                [0, [Tile(132, True)]],
            ]

        table = MahjongTable(player1, player2, player3, player4)
        table.round = 1
        table.honba = 1
        table.riichi = 1
        table.last_player = 3
        table.last_action = 1

        self.table = table

    def is_num_of_tiles_ok(self):
        for p in self.table.players:
            for hands_info in p.hands_area:
                num_of_tiles = len(hands_info[1])
            for opens_info in p.opens_area:
                if 0 <= opens_info[0] <= 3:
                    num_of_tiles += len(opens_info[1])
                elif 4 <= opens_info[0] <= 7:
                    num_of_tiles += len(opens_info[1]) - 1

            if num_of_tiles < 13 or 14 < num_of_tiles:
                print("ERROR: The number of tiles is inaccurate.")
                print("player:", p.player_idx, num_of_tiles)
                return False
        return True

    def add_status(self, tiles: List, is_opens=False):
        result = []
        for x, tiles in tiles:
            if is_opens:
                result.append("".join([tile.char for tile in tiles]) + get_modifier(x))
            else:
                result.append(" ".join([tile.char for tile in tiles]) + get_modifier(x))
        if is_opens:
            return ", ".join(result)
        return " ".join(result)

    def split_discards(self, discards):
        return [discards[idx : idx + 6] for idx in range(0, len(discards), 6)]

    def show_all(self) -> None:
        if not self.is_num_of_tiles_ok():
            exit(1)

        print()
        print(
            get_wind_char(self.table.round // 4) + str(self.table.round % 4 + 1) + "局",
            end="",
        )
        if self.table.honba > 0:
            print(" " + str(self.table.honba) + "本場", end="")
        if self.table.riichi > 0:
            print(" " + "供託" + str(self.table.riichi))
        print()

        self.table.players.sort(key=lambda x: x.player_idx)

        for p in self.table.players:
            if p.player_idx == 0:
                print("起家")
            print(
                get_wind_char(p.wind),
                "[",
                "".join([str(p.score) + (", リーチ" if p.is_riichi else "")]),
                "]",
            )
            print(
                self.add_status(p.hands_area)
                + ", "
                + self.add_status(p.opens_area, True)
            )
            discards = self.split_discards(p.discard_area)
            print("\n".join([self.add_status(tiles) for tiles in discards]),)
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
