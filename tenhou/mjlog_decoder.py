from typing import List, Tuple, Dict, Iterator
import xml.etree.ElementTree as ET
import json
import urllib.parse
from xml.etree.ElementTree import Element
from google.protobuf import json_format

import mahjong_pb2


class MjlogParser:
    def __init__(self):
        pass

    def parse(self, path_to_mjlog: str) -> Iterator[mahjong_pb2.State]:
        tree = ET.parse(path_to_mjlog)
        root = tree.getroot()
        yield from self._parse_each_game(root)

    def _parse_each_game(self, root: Element) -> Iterator[mahjong_pb2.State]:
        assert (root.tag == "mjloggm")
        assert (root.attrib['ver'] == "2.3")
        shuffle = root.iter("SHUFFLE")
        go = root.iter("GO")
        un = root.iter("UN")
        # print(urllib.parse.unquote(child.attrib["n0"]))
        taikyoku = root.iter("TAIKYOKU")

        kv: List[Tuple[str, Dict[str, str]]] = []
        for child in root:
            if child.tag in ["SHUFFLE", "GO", "UN", "TAIKYOKU"]:
                continue
            if child.tag == "INIT":
                if kv:
                    yield from self._parse_each_round(kv)
                kv = []
            kv.append((child.tag, child.attrib))
        if kv:
            yield from self._parse_each_round(kv)

    def _parse_each_round(self, kv: List[Tuple[str, Dict[str, str]]]) -> Iterator[mahjong_pb2.State]:
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
        - <RYUUKYOKU ba="0,1" sc="250,-10,240,30,250,-10,250,-10" hai1="43,47,49,51,52,54,56,57,62,79,82,101,103" />
          - key = "RYUUKYOKU" val = {'ba': '0,1', 'sc': '250,-10,240,30,250,-10,250,-10', 'hai1': '43,47,49,51,52,54,56,57,62,79,82,101,103'}
        - <AGARI ba="1,3" hai="1,6,9,24,25,37,42,44,45,49,52,58,60,64" machi="44" ten="30,8000,1" yaku="1,1,7,1,52,1,54,1,53,1" doraHai="69" doraHaiUra="59" who="2" fromWho="3" sc="240,0,260,0,230,113,240,-83" />
          - key = "AGARI" val = {'ba': '1,3', 'hai': '1,6,9,24,25,37,42,44,45,49,52,58,60,64', 'machi': '44', 'ten': '30,8000,1', 'yaku': '1,1,7,1,52,1,54,1,53,1', 'doraHai': '69', 'doraHaiUra': '59', 'who': '2', 'fromWho': '3', 'sc': '240,0,260,0,230,113,240,-83'}
        """
        print(kv[0])
        for key, val in kv:
            yield mahjong_pb2.State()


if __name__ == "__main__":
    # print(json.dumps(json_format.MessageToDict(s)))
    parser = MjlogParser()
    for state in parser.parse("resources/2011020417gm-00a9-0000-b67fcaa3.mjlog"):
        pass

