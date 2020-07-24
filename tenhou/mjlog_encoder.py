import json
import urllib.parse
from google.protobuf import json_format

import mahjong_pb2


class MjlogEncoder:
    def __init__(self):
        self.xml = ""

    def parse(self, path_to_json) -> str:
        self.xml = """<mjloggm ver="2.3"><SHUFFLE seed="" ref=""/><GO type="169" lobby="0"/>"""
        is_init_round = True
        with open(path_to_json, 'r') as fp:
            for line in fp:
                d = json.loads(line)
                state = json_format.ParseDict(d, mahjong_pb2.State())
                if is_init_round:
                    self.xml += MjlogEncoder._parse_player_id(state)
                    self.xml += """<TAIKYOKU oya="0"/>"""
                    is_init_round = False
                self.xml += self._parse_each_round(state)
        self.xml += """</mjloggm>"""
        return self.xml

    def _parse_each_round(self, state: mahjong_pb2.State) -> str:
        ret = "<INIT "
        ret += f"seed=\"{state.init_score.round},{state.init_score.honba},{state.init_score.riichi},,,{state.dora[0]}\" "
        ret += f"ten=\"{state.init_score.ten[0]},{state.init_score.ten[1]},{state.init_score.ten[2]},{state.init_score.ten[3]}\" oya=\"{state.init_score.round % 4}\" "
        hai = [",".join([str(t) for t in x.tiles]) for x in state.initial_hands]
        ret += f"hai0=\"{hai[0]}\" "
        ret += f"hai1=\"{hai[1]}\" "
        ret += f"hai2=\"{hai[2]}\" "
        ret += f"hai3=\"{hai[3]}\" "

        ret += "/>"
        return ret

    @staticmethod
    def _parse_player_id(state: mahjong_pb2.State) -> str:
        players = [urllib.parse.quote(player) for player in state.player_ids]
        return f"<UN n0=\"{players[0]}\" n1=\"{players[1]}\" n2=\"{players[2]}\" n3=\"{players[3]}\"/>"


if __name__ == "__main__":
    path_to_json = "resources/json/2011020417gm-00a9-0000-b67fcaa3.json"
    encoder = MjlogEncoder()
    with open("/Users/sotetsuk/Desktop/mjlog_test/test.mjlog", "w") as f:
        s = encoder.parse(path_to_json)
        f.write(s)
        print(s)
