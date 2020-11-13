from typing import List, Tuple, Dict, Iterator
import os
import hashlib
import json
import copy
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from google.protobuf import json_format

from mjconvert import mj_pb2


SEED_CACHE_DIR = os.path.join(os.environ["HOME"], ".mjconvert/seed_cache")


class MjlogDecoder:
    def __init__(self, modify: bool):
        self.state = None
        self.modify = modify

    def decode(self, mjlog_str: str) -> List[mj_pb2.State]:
        wall_dices = reproduce_wall_from_mjlog(mjlog_str)
        root = ET.fromstring(mjlog_str)
        ret = []
        for state in self._parse_each_game(root, wall_dices, self.modify):
            # No spaces
            x = json.dumps(json_format.MessageToDict(state, including_default_value_fields=False), ensure_ascii=False, separators=(',', ':')) + "\n"
            ret.append(x)
        return ret

    def _parse_each_game(self, root: Element, wall_dices: List[Tuple[List[int], List[int]]], modify: bool) -> Iterator[mj_pb2.State]:
        state_ = mj_pb2.State()

        assert root.tag == "mjloggm"
        assert root.attrib['ver'] == "2.3"

        shuffle = root.iter("SHUFFLE")
        go = root.iter("GO")
        for child in go:
            assert int(child.attrib["type"]) == 169  # only use 鳳南赤
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
            if child.tag in ["SHUFFLE", "GO", "UN", "TAIKYOKU", "BYE"]:
                continue
            if child.tag == "INIT":
                if kv:
                    wall, dices = wall_dices[i]
                    yield from self._parse_each_round(kv, wall, dices, modify)
                    i += 1
                self.state = copy.deepcopy(state_)
                kv = []
            kv.append((child.tag, child.attrib))
        if kv:
            wall, dices = wall_dices[i]
            yield from self._parse_each_round(kv, wall, dices, modify)
            i += 1

    def _parse_each_round(self, kv: List[Tuple[str, Dict[str, str]]], wall: List[int], dices: List[int], modify: bool) -> Iterator[mj_pb2.State]:
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
        self.state.init_score.ten[:] = [int(x) * 100 for x in val["ten"].split(",")]
        self.state.terminal.final_score.round = round_
        self.state.terminal.final_score.honba = honba
        self.state.terminal.final_score.riichi = riichi
        self.state.terminal.final_score.ten[:] = [int(x) * 100 for x in val["ten"].split(",")]
        self.state.wall[:] = wall
        self.state.doras.append(dora)
        self.state.ura_doras.append(wall[131])
        assert dora == wall[130]
        for i in range(4):
            self.state.private_infos.append(
                mj_pb2.PrivateInfo(who=i,
                                        init_hand=[int(x) for x in val["hai" + str(i)].split(",")])
            )
        for i in range(4 * 12):
            assert wall[i] in self.state.private_infos[((i // 4) + round_) % 4].init_hand
        for i in range(4 * 12, 4 * 13):
            assert wall[i] in self.state.private_infos[(i + round_) % 4].init_hand

        event = None
        num_kan_dora = 0
        last_drawer, last_draw = None, None
        reach_terminal = False
        for key, val in kv[1:]:
            if key != "UN" and key[0] in ["T", "U", "V", "W"]:  # draw
                who = MjlogDecoder._to_absolute_pos(key[0])
                draw = int(key[1:])
                self.state.private_infos[int(who)].draws.append(draw)
                event = mj_pb2.Event(
                    who=who,
                    type=mj_pb2.EVENT_TYPE_DRAW,
                    # tile is set empty because this is private information
                )
                last_drawer, last_draw = who, draw
            elif key != "DORA" and key[0] in ["D", "E", "F", "G"]:  # discard
                who = MjlogDecoder._to_absolute_pos(key[0])
                discard = int(key[1:])
                type_ = mj_pb2.EVENT_TYPE_DISCARD_FROM_HAND
                if last_drawer is not None and last_draw is not None and last_drawer == who and last_draw == discard:
                    type_ = mj_pb2.EVENT_TYPE_DISCARD_DRAWN_TILE
                event = mj_pb2.Event(
                    who=who,
                    type=type_,
                    tile=discard,
                )
                last_drawer, last_draw = None, None
            elif key == "N":  # open
                who = int(val["who"])
                open = int(val["m"])
                event = mj_pb2.Event(
                    who=who,
                    type=MjlogDecoder._open_type(open),
                    open=open,
                )
            elif key == "REACH":
                who = int(val["who"])
                if int(val["step"]) == 1:
                    event = mj_pb2.Event(
                        who=who,
                        type=mj_pb2.EVENT_TYPE_RIICHI
                    )
                else:
                    event = mj_pb2.Event(
                        who=who,
                        type=mj_pb2.EVENT_TYPE_RIICHI_SCORE_CHANGE
                    )
                    self.state.terminal.final_score.riichi += 1
                    self.state.terminal.final_score.ten[who] -= 1000
            elif key == "DORA":
                dora = wall[128 - 2 * num_kan_dora]
                assert dora == int(val["hai"])
                ura_dora = wall[129 - 2 * num_kan_dora]
                num_kan_dora += 1
                self.state.doras.append(dora)
                self.state.ura_doras.append(ura_dora)
                event = mj_pb2.Event(
                    type=mj_pb2.EVENT_TYPE_NEW_DORA,
                    tile=dora
                )
            elif key == "RYUUKYOKU":
                reach_terminal = True
                ba, riichi = [int(x) for x in val["ba"].split(",")]
                self.state.terminal.no_winner.ten_changes[:] = [int(x) * 100 for i, x in enumerate(val["sc"].split(",")) if i % 2 == 1]
                for i in range(4):
                    self.state.terminal.final_score.ten[i] += self.state.terminal.no_winner.ten_changes[i]
                for i in range(4):
                    hai_key = "hai" + str(i)
                    if hai_key not in val:
                        continue
                    self.state.terminal.no_winner.tenpais.append(mj_pb2.TenpaiHand(
                            who=i,
                            closed_tiles=[int(x) for x in val[hai_key].split(",")],
                    ))
                if "type" in val:
                    no_winner_type = None
                    if val["type"] == "yao9":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_KYUUSYU
                    elif val["type"] == "reach4":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_FOUR_RIICHI
                    elif val["type"] == "ron3":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_THREE_RONS
                    elif val["type"] == "kan4":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_FOUR_KANS
                    elif val["type"] == "kan4":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_FOUR_KANS
                    elif val["type"] == "kaze4":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_FOUR_WINDS
                    elif val["type"] == "nm":
                        no_winner_type = mj_pb2.NO_WINNER_TYPE_NM
                    assert no_winner_type is not None
                    self.state.terminal.no_winner.type = no_winner_type
                if "owari" in val:
                    # オーラス流局時のリーチ棒はトップ総取り
                    # TODO: 同着トップ時には上家が総取りしてるが正しい？
                    # TODO: 上家総取りになってない。。。
                    if self.state.terminal.final_score.riichi != 0:
                        max_ten = max(self.state.terminal.final_score.ten)
                        for i in range(4):
                            if self.state.terminal.final_score.ten[i] == max_ten:
                                self.state.terminal.final_score.ten[i] += 1000 * self.state.terminal.final_score.riichi
                                break
                    self.state.terminal.final_score.riichi = 0
                    self.state.terminal.is_game_over = True
                event = mj_pb2.Event(
                    type=mj_pb2.EVENT_TYPE_NO_WINNER
                )
            elif key == "AGARI":
                reach_terminal = True
                ba, riichi = [int(x) for x in val["ba"].split(",")]
                who = int(val["who"])
                from_who = int(val["fromWho"])
                # set event
                event = mj_pb2.Event(
                    who=who,
                    type=mj_pb2.EVENT_TYPE_TSUMO if who == from_who else mj_pb2.EVENT_TYPE_RON,
                    tile=int(val["machi"])
                )
                # set win info
                # TODO(sotetsuk): yakuman
                # TODO(sotetsuk): check double ron behavior
                win = mj_pb2.Win(
                    who=who,
                    from_who=from_who,
                    closed_tiles=[int(x) for x in val["hai"].split(",")],
                    win_tile=int(val["machi"])
                )
                win.ten_changes[:] = [int(x) * 100 for i, x in enumerate(val["sc"].split(",")) if i % 2 == 1]
                for i in range(4):
                    self.state.terminal.final_score.ten[i] += win.ten_changes[i]
                self.state.terminal.final_score.riichi = 0
                if "m" in val:
                    win.opens[:] = [int(x) for x in val["m"].split(",")]
                assert self.state.doras == [int(x) for x in val["doraHai"].split(",")]
                if "doraHaiUra" in val:
                    assert self.state.ura_doras == [int(x) for x in val["doraHaiUra"].split(",")]
                win.fu, win.ten, _ = [int(x) for x in val["ten"].split(",")]
                if modify and "yakuman" in val:
                    win.fu = 0
                if "yaku" in val:
                    assert "yakuman" not in val
                    yakus = [int(x) for i, x in enumerate(val["yaku"].split(",")) if i % 2 == 0]
                    fans = [int(x) for i, x in enumerate(val["yaku"].split(",")) if i % 2 == 1]
                    yaku_fan = [(yaku, fan) for yaku, fan in zip(yakus, fans)]
                    if modify:
                        yaku_fan.sort(key=lambda x: x[0])
                    win.yakus[:] = [x[0] for x in yaku_fan]
                    win.fans[:] = [x[1] for x in yaku_fan]
                if "yakuman" in val:
                    assert "yaku" not in val
                    yakumans = [int(x) for i, x in enumerate(val["yakuman"].split(","))]
                    if modify:
                        yakumans.sort()
                    win.yakumans[:] = yakumans
                self.state.terminal.wins.append(win)
                if "owari" in val:
                    self.state.terminal.is_game_over = True
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

        if not reach_terminal:
            self.state.ClearField("terminal")
        else:
            assert sum(self.state.terminal.final_score.ten) + self.state.terminal.final_score.riichi * 1000 == 100000

        yield copy.deepcopy(self.state)

    @staticmethod
    def _to_absolute_pos(pos_str: str) -> mj_pb2.AbsolutePos:
        assert pos_str in ["T", "U", "V", "W", "D", "E", "F", "G"]
        if pos_str in ["T", "D"]:
            return mj_pb2.ABSOLUTE_POS_INIT_EAST
        elif pos_str in ["U", "E"]:
            return mj_pb2.ABSOLUTE_POS_INIT_SOUTH
        elif pos_str in ["V", "F"]:
            return mj_pb2.ABSOLUTE_POS_INIT_WEST
        elif pos_str in ["W", "G"]:
            return mj_pb2.ABSOLUTE_POS_INIT_NORTH

    @staticmethod
    def _open_type(bits: int) -> mj_pb2.EventType:
        if 1 << 2 & bits:
            return mj_pb2.EVENT_TYPE_CHI
        elif 1 << 3 & bits:
            return mj_pb2.EVENT_TYPE_PON
        elif 1 << 4 & bits:
            return mj_pb2.EVENT_TYPE_KAN_ADDED
        else:
            if mj_pb2.RELATIVE_POS_SELF == bits & 3:
                return mj_pb2.EVENT_TYPE_KAN_CLOSED
            else:
                return mj_pb2.EVENT_TYPE_KAN_OPENED


def reproduce_wall_from_mjlog(mjlog_str: str, use_cache=False) -> List[Tuple[List[int], List[int]]]:
    # SeedをXMLから切り出す
    root = ET.fromstring(mjlog_str)
    shuffle = root.iter("SHUFFLE")
    seed = ""
    for i, child in enumerate(shuffle):
        assert i == 0
        x = child.attrib["seed"].split(",")
        assert x[0] == "mt19937ar-sha512-n288-base64"
        assert len(x) == 2
        seed = repr(x[1])[1:-1]
    assert len(seed) != 0, "Old (~2009.xx) log does not have SHUFFLE item"

    return reproduce_wall_from_seed(seed, use_cache=use_cache)


def reproduce_wall_from_seed(seed: str, use_cache=False) -> List[Tuple[List[int], List[int]]]:
    """牌山の情報をSeedから復元する。のキャッシュがあれば、それを返す

    >>> seed = "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"
    >>> wall_dices = reproduce_wall_from_seed(seed, True)
    >>> len(wall_dices)
    100
    >>> wall_dices[0]
    ([52, 72, 106, 73, 43, 62, 33, 89, 38, 54, 44, 2, 90, 59, 110, 107, 1, 61, 98, 108, 11, 0, 77, 134, 48, 15, 112, 22, 102, 101, 10, 12, 4, 126, 84, 66, 83, 120, 71, 50, 64, 53, 78, 86, 19, 34, 13, 29, 40, 81, 3, 121, 51, 129, 24, 18, 119, 21, 132, 45, 42, 114, 135, 91, 49, 105, 93, 75, 116, 74, 41, 79, 60, 30, 46, 28, 70, 131, 100, 31, 113, 133, 7, 111, 99, 14, 36, 97, 58, 76, 94, 39, 5, 65, 25, 9, 23, 68, 47, 82, 17, 16, 117, 26, 63, 32, 88, 109, 85, 55, 96, 103, 56, 123, 6, 35, 128, 20, 118, 69, 130, 92, 57, 8, 95, 115, 67, 104, 87, 125, 127, 80, 27, 37, 122, 124], [4, 5])
    """
    #
    os.makedirs(SEED_CACHE_DIR, exist_ok=True)
    seed_md5 = hashlib.md5(seed.encode()).hexdigest()
    seed_cache = os.path.join(SEED_CACHE_DIR, seed_md5 + ".txt")
    out: List[str]
    if os.path.exists(seed_cache):
        with open(seed_cache, "r") as f:
            out = f.readlines()
    else:
        # TODO: remove docker dependency
        tmp = subprocess.run(["docker", "run", "--rm", "sotetsuk/twr:v0.0.1", "/twr",  seed, "100"], capture_output=True)
        assert tmp.returncode == 0, "Failed to decode wall from given seed"
        out = tmp.stdout.decode('utf-8').strip('\n').split('\n')
        if use_cache:
            with open(seed_cache, "w") as f:
                for line in out:
                    f.write(line + "\n")

    return parse_wall(out)


def parse_wall(wall_outputs: List[str]) -> List[Tuple[List[int], List[int]]]:
    """牌山の前処理をする

    >>> wall_outputs = ["124,122,37,27,80,127,125,87,104,67,115,95,8,57,92,130,69,118,20,128,35,6,123,56,103,96,55,85,109,88,32,63,26,117,16,17,82,47,68,23,9,25,65,5,39,94,76,58,97,36,14,99,111,7,133,113,31,100,131,70,28,46,30,60,79,41,74,116,75,93,105,49,91,135,114,42,45,132,21,119,18,24,129,51,121,3,81,40,29,13,34,19,86,78,53,64,50,71,120,83,66,84,126,4,12,10,101,102,22,112,15,48,134,77,0,11,108,98,61,1,107,110,59,90,2,44,54,38,89,33,62,43,73,106,72,52", "4,5"]
    >>> wall_dices = parse_wall(wall_outputs)
    >>> len(wall_dices)
    1
    >>> wall_dices
    [([52, 72, 106, 73, 43, 62, 33, 89, 38, 54, 44, 2, 90, 59, 110, 107, 1, 61, 98, 108, 11, 0, 77, 134, 48, 15, 112, 22, 102, 101, 10, 12, 4, 126, 84, 66, 83, 120, 71, 50, 64, 53, 78, 86, 19, 34, 13, 29, 40, 81, 3, 121, 51, 129, 24, 18, 119, 21, 132, 45, 42, 114, 135, 91, 49, 105, 93, 75, 116, 74, 41, 79, 60, 30, 46, 28, 70, 131, 100, 31, 113, 133, 7, 111, 99, 14, 36, 97, 58, 76, 94, 39, 5, 65, 25, 9, 23, 68, 47, 82, 17, 16, 117, 26, 63, 32, 88, 109, 85, 55, 96, 103, 56, 123, 6, 35, 128, 20, 118, 69, 130, 92, 57, 8, 95, 115, 67, 104, 87, 125, 127, 80, 27, 37, 122, 124], [4, 5])]
    """
    wall_dices: List[Tuple[List[int], List[int]]] = []
    wall, dices = [], []
    for i, line in enumerate(wall_outputs):
        line_splitted = line.strip().strip("\n").split(',')
        if i % 2 == 0:
            wall = [int(x) for x in line_splitted]
            wall.reverse()
            assert len(wall) == 136
        else:
            dices = [int(x) for x in line_splitted]
            assert len(dices) == 2
            wall_dices.append((wall, dices))

    return wall_dices
