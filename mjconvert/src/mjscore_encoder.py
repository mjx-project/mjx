import os
from typing import List
import copy
import json
import urllib.parse
from google.protobuf import json_format

import mj_pb2


# ここを実装
def mjproto_to_mjscore(state: mj_pb2.State) -> str:
    #print(state.init_score.round)
    #print(state.private_infos.ABSOLUTE_POS_INIT_EAST.init_hand)
    # print(state.init_score.honba)
    #print(state.init_score.ten)
    #print(state.private_infos)
    round:int = state.init_score.round
    honba:int = state.init_score.honba
    riichi:int = state.init_score.riichi
    doras:List[int] = [i for i in state.doras]
    ura_doras:List[int] = [i for i in state.ura_doras]
    init_score:List[int] = [i for i in state.init_score.ten]

    d = {'title':[], 'name':[], 'rule':[],'log':[[[[round,honba,riichi],init_score,doras,ura_doras]]]}
    return json.dumps(d)


if __name__ == '__main__':
    # 東1局0本場の mjproto
    path_to_mjproto_example = "../..//test/resources/json/first-example.json"
    with open(path_to_mjproto_example, 'r') as f:
        line = f.readline()
    d = json.loads(line)
    state: mj_pb2.State = json_format.ParseDict(d, mj_pb2.State())

    # 東1局0本場の mjscore
    path_to_mjscore_example = "../../test/resources/mjscore/first-example.json"
    with open(path_to_mjscore_example, 'r') as f:
        line = f.readline()
    mjscore_expected_dict = json.loads(line)

    # 実装を使って変換
    mjscore_str = mjproto_to_mjscore(state)
    mjscore_dict = json.loads(mjscore_str)

    # 比較
    print(mjscore_expected_dict)
    print(mjscore_dict)
    print(mjscore_expected_dict == mjscore_dict)

