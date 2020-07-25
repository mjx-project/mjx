from typing import List, Tuple, Dict, Iterator
import os
import json
import copy
import argparse
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from google.protobuf import json_format

import mahjong_pb2


parser = argparse.ArgumentParser(description="""Convert Tenhou's mjlog format into json, which is readable as protocol buffer.

Example:

  $ python mjlog_decoder.py resources/mjlog resources/json
""")
parser.add_argument('mjlog_dir', help='Path to mjlogs')    # 必須の引数を追加
parser.add_argument('json_dir', help='Path to json outputs')    # 必須の引数を追加
args = parser.parse_args()


class MjlogDecoder:
    def __init__(self):
        self.state = None

    def parse(self, path_to_mjlog: str, wall_dices: List[Tuple[List[int], List[int]]]) -> Iterator[mahjong_pb2.State]:
        tree = ET.parse(path_to_mjlog)
        root = tree.getroot()
        yield from self._parse_each_game(root, wall_dices)

    def _parse_each_game(self, root: Element, wall_dices: List[Tuple[List[int], List[int]]]) -> Iterator[mahjong_pb2.State]:
        state_ = mahjong_pb2.State()

        assert root.tag == "mjloggm"
        assert root.attrib['ver'] == "2.3"

        shuffle = root.iter("SHUFFLE")
        go = root.iter("GO")
        un = root.iter("UN")  # TODO(sotetsuk): if there are > 2 "UN", some user became offline
        for child in un:
            state_.player_ids.append(urllib.parse.unquote(child.attrib["n0"]))
            state_.player_ids.append(urllib.parse.unquote(child.attrib["n1"]))
            state_.player_ids.append(urllib.parse.unquote(child.attrib["n2"]))
            state_.player_ids.append(urllib.parse.unquote(child.attrib["n3"]))
            break
        taikyoku = root.iter("TAIKYOKU")

        kv: List[Tuple[str, Dict[str, str]]] = []
        i = 0
        for child in root:
            if child.tag in ["SHUFFLE", "GO", "UN", "TAIKYOKU"]:
                continue
            if child.tag == "INIT":
                if kv:
                    wall, dices = wall_dices[i]
                    yield from self._parse_each_round(kv, wall, dices)
                    i += 1
                self.state = copy.deepcopy(state_)
                kv = []
            kv.append((child.tag, child.attrib))
        if kv:
            wall, dices = wall_dices[i]
            yield from self._parse_each_round(kv, wall, dices)
            i += 1

    def _parse_each_round(self, kv: List[Tuple[str, Dict[str, str]]], wall: List[int], dices: List[int]) -> Iterator[mahjong_pb2.State]:
        """Input examples

        - <INIT seed="0,0,0,2,2,112" ten="250,250,250,250" oya="0" hai0="48,16,19,34,2,76,13,7,128,1,39,121,87" hai1="17,62,79,52,56,57,82,98,32,103,24,70,54" hai2="55,30,12,26,31,90,3,4,80,125,66,102,78" hai3="120,130,42,67,114,93,5,61,20,108,41,100,84"/>
          - key = INIT val = {'seed': '0,0,0,2,2,112', 'ten': '250,250,250,250', 'oya': '0', 'hai0': '48,16,19,34,2,76,13,7,128,1,39,121,87', 'hai1': '17,62,79,52,56,57,82,98,32,103,24,70,54', 'hai2': '55,30,12,26,31,90,3,4,80,125,66,102,78', 'hai3': '120,130,42,67,114,93,5,61,20,108,41,100,84'}
        - <F37/>:
          - key = "F37", val = ""
        - <REACH who="1" step="1"/>
          - key = "REACH", val = {'who': '1', 'step': '1'}
        - <REACH who="1" ten="250,240,250,250" step="2"/>
          - key = "REACH", val = {'who': '1', 'ten': '250,240,250,250', 'step': '2'}
        - <N who="3" m="42031" />
          - key = "N", val = {'who': '3', 'm': '42031'}
        - <DORA hai="8" />
          - key = "DORA", val = {'hai': '8'}
        - <RYUUKYOKU ba="0,1" sc="250,-10,240,30,250,-10,250,-10" hai1="43,47,49,51,52,54,56,57,62,79,82,101,103" />
          - key = "RYUUKYOKU" val = {'ba': '0,1', 'sc': '250,-10,240,30,250,-10,250,-10', 'hai1': '43,47,49,51,52,54,56,57,62,79,82,101,103'}
        - <AGARI ba="1,3" hai="1,6,9,24,25,37,42,44,45,49,52,58,60,64" machi="44" ten="30,8000,1" yaku="1,1,7,1,52,1,54,1,53,1" doraHai="69" doraHaiUra="59" who="2" fromWho="3" sc="240,0,260,0,230,113,240,-83" />
          - key = "AGARI" val = {'ba': '1,3', 'hai': '1,6,9,24,25,37,42,44,45,49,52,58,60,64', 'machi': '44', 'ten': '30,8000,1', 'yaku': '1,1,7,1,52,1,54,1,53,1', 'doraHai': '69', 'doraHaiUra': '59', 'who': '2', 'fromWho': '3', 'sc': '240,0,260,0,230,113,240,-83'}
        """
        key, val = kv[0]
        assert key == "INIT"
        round_, honba, riichi, dice1, dice2, dora = [int(x) for x in val["seed"].split(",")]
        self.state.init_score.round = round_
        self.state.init_score.honba = honba
        self.state.init_score.riichi = riichi
        self.state.init_score.ten[:] = [int(x) for x in val["ten"].split(",")]
        self.state.curr_score.round = round_
        self.state.curr_score.honba = honba
        self.state.curr_score.riichi = riichi
        self.state.curr_score.ten[:] = [int(x) for x in val["ten"].split(",")]
        self.state.wall[:] = wall
        self.state.doras.append(dora)
        self.state.ura_doras.append(wall[131])
        assert dora == wall[130]
        for i in range(4):
            self.state.init_hands.append(mahjong_pb2.InitHand(who=i, tiles=[int(x) for x in val["hai" + str(i)].split(",")]))
        for i in range(4 * 12):
            assert wall[i] in self.state.init_hands[((i // 4) + round_) % 4].tiles
        for i in range(4 * 12, 4 * 13):
            assert wall[i] in self.state.init_hands[(i + round_ ) % 4].tiles

        last_drawer, last_draw = None, None
        event = None
        num_kan_dora = 0
        for key, val in kv[1:]:
            if key[0] in ["T", "U", "V", "W"]:  # draw
                # TODO (sotetsuk): consider draw after kan case
                who = MjlogDecoder._to_wind(key[0])
                draw = int(key[1:])
                event = mahjong_pb2.Event(
                    who=who,
                    type=mahjong_pb2.EVENT_TYPE_DRAW,
                    tile=draw,
                )
                last_drawer, last_draw = who, draw
            elif key[0] in ["D", "E", "F", "G"]:  # discard
                # TDOO (sotetsuk): riichi
                who = MjlogDecoder._to_wind(key[0])
                discard = int(key[1:])
                event = mahjong_pb2.Event(
                    who=who,
                    type=mahjong_pb2.EVENT_TYPE_DISCARD,
                    tile=discard,
                )
            elif key == "N":
                who = int(val["who"])
                open = int(val["m"])
                event = mahjong_pb2.Event(
                    who=who,
                    type=MjlogDecoder._open_type(open),
                    open=open,
                )
            elif key == "REACH":
                who = int(val["who"])
                if int(val["step"]) == 1:
                    event = mahjong_pb2.Event(
                        who=who,
                        type=mahjong_pb2.EVENT_TYPE_RIICHI
                    )
                else:
                    self.state.curr_score.riichi += 1
                    self.state.curr_score.ten[:] = [int(x) for x in val["ten"].split(",")]
                    event = mahjong_pb2.Event(
                        who=who,
                        type=mahjong_pb2.EVENT_TYPE_RIICHI_SCORE_CHANGE
                    )
            elif key == "DORA":
                dora = wall[129 - 2 * num_kan_dora]
                assert dora == int(val["hai"])
                ura_dora = wall[128 - 2 * num_kan_dora]
                num_kan_dora += 1
                self.state.doras.append(dora)
                self.state.ura_doras.append(ura_dora)
                event = mahjong_pb2.Event(
                    type=mahjong_pb2.EVENT_TYPE_NEW_DORA,
                    tile=dora
                )
            elif key == "RYUUKYOKU":
                ba, riichi = [int(x) for x in val["ba"].split(",")]
                assert ba == self.state.curr_score.honba
                assert riichi == self.state.curr_score.riichi
                self.state.end_info.ten_before[:] = [int(x) for i, x in enumerate(val["sc"].split(",")) if i % 2 == 0]
                self.state.end_info.ten_changes[:] = [int(x) for i, x in enumerate(val["sc"].split(",")) if i % 2 == 1]
                self.state.end_info.ten_after[:] = [x + y for x, y in zip(self.state.end_info.ten_before, self.state.end_info.ten_changes)]
                self.state.curr_score.ten[:] = self.state.end_info.ten_after[:]
                for i in range(4):
                    hai_key = "hai" + str(i)
                    if hai_key not in val:
                        continue
                    self.state.end_info.tenpais.append(mahjong_pb2.TenpaiHand(
                            who=i,
                            closed_tiles=[int(x) for x in val[hai_key].split(",")],
                    ))
                if "type" in val:
                    self.state.no_winner_end_type = val["type"]
                if "owari" in val:
                    self.state.end_info.is_game_over = True
            elif key == "AGARI":
                ba, riichi = [int(x) for x in val["ba"].split(",")]
                assert ba == self.state.curr_score.honba
                assert riichi == self.state.curr_score.riichi
                who = int(val["who"])
                from_who = int(val["fromWho"])
                # set event
                event = mahjong_pb2.Event(
                    who=who,
                    type=mahjong_pb2.EVENT_TYPE_TSUMO if who == from_who else mahjong_pb2.EVENT_TYPE_RON
                )
                # set win info
                # TODO(sotetsuk): yakuman
                # TODO(sotetsuk): check double ron behavior
                self.state.end_info.ten_before[:] = [int(x) for i, x in enumerate(val["sc"].split(",")) if i % 2 == 0]
                self.state.end_info.ten_changes[:] = [int(x) for i, x in enumerate(val["sc"].split(",")) if i % 2 == 1]
                self.state.end_info.ten_after[:] = [x + y for x, y in zip(self.state.end_info.ten_before, self.state.end_info.ten_changes)]
                self.state.curr_score.ten[:] = self.state.end_info.ten_after[:]
                win = mahjong_pb2.Win(
                    who=who,
                    from_who=from_who,
                    closed_tiles=[int(x) for x in val["hai"].split(",")],
                    win_tile=int(val["machi"])
                )
                if "m" in val:
                    win.opens[:] = [int(x) for x in val["m"].split(",")]
                assert self.state.doras == [int(x) for x in val["doraHai"].split(",")]
                if "doraHaiUra" in val:
                    assert self.state.ura_doras == [int(x) for x in val["doraHaiUra"].split(",")]
                win.fu, win.ten, _ = [int(x) for x in val["ten"].split(",")]
                win.yakus[:] = [int(x) for i, x in enumerate(val["yaku"].split(",")) if i % 2 == 0]
                win.fans[:] = [int(x) for i, x in enumerate(val["yaku"].split(",")) if i % 2 == 1]
                if "yakuman" in val:
                    win.yakumans[:] = [int(x) for i, x in enumerate(val["yakuman"].split(",")) if i % 2 == 0]
                self.state.end_info.wins.append(win)
                if "owari" in val:
                    self.state.end_info.is_game_over = True
            elif key == "BYE":  # 接続切れ
                pass
            elif key == "UN":  # 再接続
                pass
            else:
                raise KeyError(key)

            if event is not None:
                self.state.event_history.events.append(event)
            event = None
            # yield copy.deepcopy(self.state)

        yield copy.deepcopy(self.state)

    @staticmethod
    def _to_wind(wind_str: str) -> mahjong_pb2.Wind:
        assert wind_str in ["T", "U", "V", "W", "D", "E", "F", "G"]
        if wind_str in ["T", "D"]:
            return mahjong_pb2.WIND_EAST
        elif wind_str in ["U", "E"]:
            return mahjong_pb2.WIND_SOUTH
        elif wind_str in ["V", "F"]:
            return mahjong_pb2.WIND_WEST
        elif wind_str in ["W", "G"]:
            return mahjong_pb2.WIND_NORTH

    @staticmethod
    def _open_type(bits: int) -> mahjong_pb2.EventType:
        if 1 << 2 & bits:
            return mahjong_pb2.EVENT_TYPE_CHI
        elif 1 << 3 & bits:
            if 1 << 4 & bits:
                return mahjong_pb2.EVENT_TYPE_KAN_ADDED
            else:
                return mahjong_pb2.EVENT_TYPE_PON
        else:
            if mahjong_pb2.RelativePos(bits & 3) == mahjong_pb2.RELATIVE_POS_SELF:
                return mahjong_pb2.EVENT_TYPE_KAN_CLOSED
            else:
                return mahjong_pb2.EVENT_TYPE_KAN_OPENED


# TODO: remove docker dependency
def reproduce_wall(path_to_mjlog: str) -> List[Tuple[List[int], List[int]]]:
    tree = ET.parse(path_to_mjlog)
    root = tree.getroot()
    shuffle = root.iter("SHUFFLE")
    seed = ""
    for i, child in enumerate(shuffle):
        assert i == 0
        x = child.attrib["seed"].split(",")
        assert x[0] == "mt19937ar-sha512-n288-base64"
        assert len(x) == 2
        seed = repr(x[1])[1:-1]
    out = subprocess.run(["docker", "run", "sotetsuk/twr:v0.0.1", "/twr",  seed, "100"], capture_output=True)
    wall_dices: List[Tuple[List[int], List[int]]] = []
    wall, dices = [], []
    for i, line in enumerate(out.stdout.decode('utf-8').strip('\n').split('\n')):
        if i % 2 == 0:
            wall = [int(x) for x in line.split(',')]
            wall.reverse()
            assert len(wall) == 136
        else:
            dices = [int(x) for x in line.split(',')]
            assert len(dices) == 2
            wall_dices.append((wall, dices))
    return wall_dices


if __name__ == "__main__":
    parser = MjlogDecoder()
    os.makedirs(args.json_dir, exist_ok=True)
    for mjlog in os.listdir(args.mjlog_dir):
        if not mjlog.endswith("mjlog"):
            continue

        path_to_mjlog = os.path.join(args.mjlog_dir, mjlog)
        path_to_json = os.path.join(args.json_dir, os.path.splitext(os.path.basename(path_to_mjlog))[0] + '.json')

        print(f"converting:\t{path_to_mjlog}")
        wall_dices = reproduce_wall(path_to_mjlog)
        with open(path_to_json, 'w') as f:
            for state in parser.parse(path_to_mjlog, wall_dices):
                f.write(json.dumps(json_format.MessageToDict(state, including_default_value_fields=False), ensure_ascii=False) + "\n")
        print(f"done:\t{path_to_json}")
