from typing import List, Tuple, Dict, Iterator
import json
import copy
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from google.protobuf import json_format

import mahjong_pb2


class MjlogParser:
    def __init__(self):
        self.state = None

    def parse(self, path_to_mjlog: str, wall_dices: List[Tuple[List[int], List[int]]]) -> Iterator[mahjong_pb2.State]:
        tree = ET.parse(path_to_mjlog)
        root = tree.getroot()
        yield from self._parse_each_game(root, wall_dices)

    def _parse_each_game(self, root: Element, wall_dices: List[Tuple[List[int], List[int]]]) -> Iterator[mahjong_pb2.State]:
        assert root.tag == "mjloggm"
        assert root.attrib['ver'] == "2.3"
        shuffle = root.iter("SHUFFLE")
        go = root.iter("GO")
        un = root.iter("UN")  # TODO(sotetsuk): if there are > 2 "UN", some user became offline
        # print(urllib.parse.unquote(child.attrib["n0"]))
        taikyoku = root.iter("TAIKYOKU")

        self.state = mahjong_pb2.State()
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
                self.state = mahjong_pb2.State()
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
        self.state.dora.append(dora)
        for i in range(4):
            self.state.initial_hands.append(mahjong_pb2.InitialHand(who=i, tiles=[int(x) for x in val["hai" + str(i)].split(",")]))
        for i in range(4 * 12):
            assert wall[i] in self.state.initial_hands[((i // 4) + round_) % 4].tiles
        for i in range(4 * 12, 4 * 13):
            assert wall[i] in self.state.initial_hands[(i + round_ ) % 4].tiles
        assert dora == wall[130]

        last_drawer, last_draw = None, None
        taken_action = None
        for key, val in kv[1:]:
            if key[0] in ["T", "U", "V", "W"]:  # draw
                # TODO (sotetsuk): consider draw after kan case
                who = MjlogParser._to_wind(key[0])
                draw = int(key[1:])
                taken_action = mahjong_pb2.TakenAction(
                    who=who,
                    type=mahjong_pb2.ACTION_TYPE_DRAW,
                    draw=draw,
                )
                last_drawer, last_draw = who, draw
            elif key[0] in ["D", "E", "F", "G"]:  # discard
                # TDOO (sotetsuk): riichi
                who = MjlogParser._to_wind(key[0])
                discard = int(key[1:])
                discard_drawn_tile = last_drawer == who and last_draw == discard
                taken_action = mahjong_pb2.TakenAction(
                    who=who,
                    type=mahjong_pb2.ACTION_TYPE_DISCARD,
                    discard=discard,
                    discard_drawn_tile=discard_drawn_tile,
                )
            elif key == "N":
                who = int(val["who"])
                open = int(val["m"])
                taken_action = mahjong_pb2.TakenAction(
                    who=who,
                    type=MjlogParser._open_type(open),
                    open=open,
                )
                continue
            elif key == "REACH":
                if int(val["step"]) == 1:
                    who = int(val["who"])
                    taken_action = mahjong_pb2.TakenAction(
                        who=who,
                        type=mahjong_pb2.ACTION_TYPE_RIICHI
                    )
                else:
                    self.state.curr_score.riichi += 1
                    self.state.curr_score.ten[:] = [int(x) for x in val["ten"].split(",")]
                    continue
            elif key == "DORA":
                dora = int(val["hai"])
                self.state.dora.append(dora)
                continue
            elif key == "RYUUKYOKU":
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
                if "owari" in val:
                    self.state.end_info.is_game_over = True
            elif key == "AGARI":
                who = int(val["who"])
                from_who = int(val["fromWho"])
                # set taken_action
                taken_action = mahjong_pb2.TakenAction(
                    who=who,
                    type=mahjong_pb2.ACTION_TYPE_TSUMO if who == from_who else mahjong_pb2.ACTION_TYPE_RON
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
                    tiles=[int(x) for x in val["hai"].split(",")],
                    win_tile=int(val["machi"])
                )
                if "m" in val:
                    win.opens[:] = [int(x) for x in val["m"].split(",")]
                win.dora[:] = [int(x) for x in val["doraHai"].split(",")]
                if "doraHaiUra" in val:
                    win.ura_dora[:] = [int(x) for x in val["doraHaiUra"].split(",")]
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
            self.state.action_history.taken_actions.append(taken_action)
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
    def _open_type(bits: int) -> mahjong_pb2.ActionType:
        if 1 << 2 & bits:
            return mahjong_pb2.ACTION_TYPE_CHI
        elif 1 << 3 & bits:
            if 1 << 4 & bits:
                return mahjong_pb2.ACTION_TYPE_KAN_ADDED
            else:
                return mahjong_pb2.ACTION_TYPE_PON
        else:
            if mahjong_pb2.RelativePos(bits & 3) == mahjong_pb2.RELATIVE_POS_SELF:
                return mahjong_pb2.ACTION_TYPE_KAN_CLOSED
            else:
                return mahjong_pb2.ACTION_TYPE_KAN_OPENED


# TODO: remove docker dependency
def reproduce_wall(path_to_mjlog: str) -> List[Tuple[List[int], List[int]]]:
    tree = ET.parse(path_to_mjlog)
    root = tree.getroot()
    shuffle = root.iter("SHUFFLE")
    seed = ""
    for i, child in enumerate(shuffle):
        assert i == 0
        x = child.attrib["seed"].split(",")
        print(x[0])
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
    parser = MjlogParser()
    path_to_mjlog = "resources/2011020417gm-00a9-0000-b67fcaa3.mjlog"
    wall_dices = reproduce_wall(path_to_mjlog)
    for state in parser.parse(path_to_mjlog, wall_dices):
        print(json.dumps(json_format.MessageToDict(state, including_default_value_fields=False)))
