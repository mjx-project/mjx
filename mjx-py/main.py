from rich.console import Console
from rich.table import Table

import mjxproto
from GetChar import get_char
from GetPos import get_pos
from GetUnicode import get_unicode

console = Console()
table = Table(show_header=True, header_style="bold magenta")


class Players:
    def __init__(self, id):
        self.id = id
        self.discards = []

    def discard(self, tile):
        self.discards.append(tile)

    def show_discard(self):
        print(self.discards)


class GameBoard:
    def __init__(self, path):
        self.path = path
        self.east = Players(0)
        self.south = Players(1)
        self.west = Players(2)
        self.north = Players(3)
        self.players = [self.east, self.south, self.west, self.north]

        self.console = Console()
        self.table = Table(show_header=True, header_style="bold magenta")

    def discard(self, id: int, tile: int):
        self.players[id].discard(tile)

    def show_discard(self):
        self.table.add_column("東", width=24, justify="center")
        self.table.add_column("南", width=24, justify="center")
        self.table.add_column("西", width=24, justify="center")
        self.table.add_column("北", width=24, justify="center")
        self.table.add_row(
            str(self.east.discards),
            str(self.south.discards),
            str(self.west.discards),
            str(self.north.discards),
        )
        self.console.print(self.table)

    def run(self):
        with open(self.path, "r", errors="ignore") as f:
            for line in f:
                test = mjxproto.State()
                test.from_json(line)

                for event in test.event_history.events:
                    self.table = Table(show_header=True, header_style="bold magenta")
                    if event.type == 1:
                        self.discard(event.who, get_char(event.tile))
                        self.show_discard()

                break


game_board = GameBoard("2010091009gm-00a9-0000-83af2648&tw=2.json")
game_board.run()
