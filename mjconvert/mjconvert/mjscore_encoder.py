from __future__ import annotations  # postpone type hint evaluation or doctest fails
import copy
import json
import os
import urllib.parse
from typing import List

from google.protobuf import json_format

import mjproto


# ここを実装
def mjproto_to_mjscore(state: mjproto.State) -> str:
    # print(state.init_score.round)
    # print(state.init_score.riichi)
    # print(state.init_score.honba)
    # print(state.init_score.ten)
    d = {}
    return json.dumps(d)


if __name__ == "__main__":
    # 東1局0本場の mjproto
    path_to_mjproto_example = "../..//test/resources/json/first-example.json"
    with open(path_to_mjproto_example, "r") as f:
        line = f.readline()
    d = json.loads(line)
    state: mjproto.State = json_format.ParseDict(d, mjproto.State())

    # 東1局0本場の mjscore
    path_to_mjscore_example = "../../test/resources/mjscore/first-example.json"
    with open(path_to_mjscore_example, "r") as f:
        line = f.readline()
    mjscore_expected_dict = json.loads(line)

    # 実装を使って変換
    mjscore_str = mjproto_to_mjscore(state)
    mjscore_dict = json.loads(mjscore_str)

    # 比較
    print(mjscore_expected_dict)
    print(mjscore_dict)
    print(mjscore_expected_dict == mjscore_dict)
