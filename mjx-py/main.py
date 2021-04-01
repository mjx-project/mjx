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
        self.hands = []
        self.discards = []
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

    def draw(self, id: int, tile: int) -> None:
        self.players[id].draw(tile)
        self.players[id].drawcount += 1

    def discard(self, id: int, tile: int) -> None:
        self.players[id].discard(tile)

    def arrange(self) -> None:
        for p in self.players:
            p.hands.sort()

    def get_dwawcount(self, id: int) -> int:
        return self.players[id].drawcount

    def show_all(self) -> None:
        if self.is_unicode:
            e_hand = [get_unicode(i).decode("unicode-escape") for i in self.east.hands]
            s_hand = [get_unicode(i).decode("unicode-escape") for i in self.south.hands]
            w_hand = [get_unicode(i).decode("unicode-escape") for i in self.west.hands]
            n_hand = [get_unicode(i).decode("unicode-escape") for i in self.north.hands]
            e_dis = [
                get_unicode(i).decode("unicode-escape") for i in self.east.discards
            ]
            s_dis = [
                get_unicode(i).decode("unicode-escape") for i in self.south.discards
            ]
            w_dis = [
                get_unicode(i).decode("unicode-escape") for i in self.west.discards
            ]
            n_dis = [
                get_unicode(i).decode("unicode-escape") for i in self.north.discards
            ]
        else:
            e_hand = [get_char(i) for i in self.east.hands]
            s_hand = [get_char(i) for i in self.south.hands]
            w_hand = [get_char(i) for i in self.west.hands]
            n_hand = [get_char(i) for i in self.north.hands]
            e_dis = [get_char(i) for i in self.east.discards]
            s_dis = [get_char(i) for i in self.south.discards]
            w_dis = [get_char(i) for i in self.west.discards]
            n_dis = [get_char(i) for i in self.north.discards]

        table_data = [
            ["      東      ", "      南      ", "      西      ", "      北      "],
            ["".join(e_hand), "".join(s_hand), "".join(w_hand), "".join(n_hand)],
            [" ".join(e_dis), " ".join(s_dis), " ".join(w_dis), " ".join(n_dis)],
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

                    if event.type == 0:
                        print(
                            f"player {event.who} draw tile {gamedata.private_infos[event.who].draws[self.get_dwawcount(event.who)]}"
                        )
                        print(
                            f"this tile is {get_char(gamedata.private_infos[event.who].draws[self.get_dwawcount(event.who)])}"
                        )
                        self.draw(
                            event.who,
                            gamedata.private_infos[event.who].draws[
                                self.get_dwawcount(event.who)
                            ],
                        )
                    elif event.type == 1:
                        print(f"player {event.who} discard tile {event.tile} from hand")
                        print(f"this tile is {get_char(event.tile)}")
                        self.discard(event.who, event.tile)
                    elif event.type == 2:
                        print(f"player {event.who} discard tile {event.tile} from draw")
                        print(f"this tile is {get_char(event.tile)}")
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
