import mjxproto
from GetUnicode import get_unicode


class Players:
    def __init__(self, id):
        self.id = id
        self.discards = []

    def discard(self, tile):
        self.discards.append(tile)

    def show_discard(self):
        print(self.id, end=":")
        print(self.discards)


with open("2010091009gm-00a9-0000-83af2648&tw=2.json", "r", errors="ignore") as f:
    for line in f:
        test = mjxproto.State()
        test.from_json(line)
        players = []
        for i in range(4):
            players.append(Players(i))

        for i, event in enumerate(test.event_history.events):
            if event.type == 1:
                players[event.who].discard(get_unicode(event.tile))
                for p in players:
                    p.show_discard()

                print("============================")

        break
