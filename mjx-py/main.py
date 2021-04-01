import argparse
import os

from terminaltables import AsciiTable, SingleTable

import mjxproto
from GetChar import get_char
from GetPos import get_pos
from GetUnicode import get_unicode


def clear_screen() -> None:
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


class Players:
    def __init__(self, id):
        self.id = id
        self.hands = []  # 手牌
        self.discards = []  # 河
        self.chi = []
        self.pon = []
        self.kan_closed = []
        self.kan_opened = []
        self.kan_added = []
        self.drawcount = 0

    def init_hand(self, init_hand) -> None:
        self.hands = init_hand

    def draw(self, tile) -> None:
        self.hands.append(tile)

    def discard(self, tile) -> None:
        if tile in self.hands:
            self.hands.remove(tile)
        self.discards.append(tile)

    def show_discard(self) -> None:
        print(self.discards)


class GameBoard:
    def __init__(self, path, is_uni):
        self.path = path
        self.is_unicode = is_uni
        self.east = Players(0)
        self.south = Players(1)
        self.west = Players(2)
        self.north = Players(3)
        self.players = [self.east, self.south, self.west, self.north]

    # idのプレイヤーがtileを取る
    def draw(self, id: int, tile: int) -> None:
        self.players[id].draw(tile)
        self.players[id].drawcount += 1

    # idのプレイヤーがtileを捨てる
    def discard(self, id: int, tile: int) -> None:
        self.players[id].discard(tile)

    # 理牌
    def arrange(self) -> None:
        for p in self.players:
            p.hands.sort()

    # drawcountは何回ツモしたかのカウンタ
    # idのプレイヤーのdwawcountを返す
    def get_dwawcount(self, id: int) -> int:
        return self.players[id].drawcount

    # 全プレイヤーの手牌、河を表示
    def show_all(self) -> None:
        hands = []
        discs = []
        for p in self.players:
            if self.is_unicode:
                hands.append([get_unicode(i).decode("unicode-escape") for i in p.hands])
                discs.append(
                    [get_unicode(i).decode("unicode-escape") for i in p.discards]
                )
            else:
                hands.append([get_char(i) for i in p.hands])
                discs.append([get_char(i) for i in p.discards])

        table_data = [
            ["      東      ", "      南      ", "      西      ", "      北      "],
            [
                "".join(hands[0]),
                "".join(hands[1]),
                "".join(hands[2]),
                "".join(hands[3]),
            ],
            [
                " ".join(discs[0]),
                " ".join(discs[1]),
                " ".join(discs[2]),
                " ".join(discs[3]),
            ],
        ]
        table_instance = SingleTable(table_data, "board")
        table_instance.inner_heading_row_border = False
        table_instance.inner_row_border = True
        table_instance.justify_columns = {
            0: "center",
            1: "center",
            2: "center",
            3: "center",
        }
        print(table_instance.table)

    # 実行
    def run(self) -> None:
        with open(self.path, "r", errors="ignore") as f:
            for line in f:
                gamedata = mjxproto.State()
                gamedata.from_json(line)

                # 手元を初期化
                for i, p in enumerate(self.players):
                    p.init_hand(gamedata.private_infos[i].init_hand)

                clear_screen()

                for i, event in enumerate(gamedata.event_history.events):

                    if event.type == 0:  # ツモ
                        print(
                            f"player {get_pos(event.who)} draw tile {gamedata.private_infos[event.who].draws[self.get_dwawcount(event.who)]}"
                        )
                        print(
                            f"{gamedata.private_infos[event.who].draws[self.get_dwawcount(event.who)]}: this tile is {get_char(gamedata.private_infos[event.who].draws[self.get_dwawcount(event.who)])}"
                        )
                        self.draw(
                            event.who,
                            gamedata.private_infos[event.who].draws[
                                self.get_dwawcount(event.who)
                            ],
                        )
                    elif event.type == 1:  # 手牌から切り
                        print(
                            f"player {get_pos(event.who)} discard tile {event.tile} from hand"
                        )
                        print(f"{event.tile}: this tile is {get_char(event.tile)}")
                        self.discard(event.who, event.tile)
                    elif event.type == 2:  # ツモから切り
                        print(
                            f"player {get_pos(event.who)} discard tile {event.tile} from draw"
                        )
                        print(f"{event.tile}: this tile is {get_char(event.tile)}")
                        self.draw(event.who, event.tile)
                        self.discard(event.who, event.tile)

                    self.arrange()
                    self.show_all()
                    input()
                    clear_screen()

                break  # 一局だけ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="2010091009gm-00a9-0000-83af2648&tw=2.json")
    parser.add_argument("--uni", default=False)
    args = parser.parse_args()

    game_board = GameBoard(args.path, args.uni)
    game_board.run()


if __name__ == "__main__":
    main()
