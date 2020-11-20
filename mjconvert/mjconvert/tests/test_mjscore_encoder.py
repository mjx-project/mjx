import os
from typing import List
import json

from mjconvert.mjlog_decoder import MjlogDecoder
from mjconvert.mjscore_encoder import mjproto_to_mjscore


def test_mjproto_to_mjscore():
    mjlog_decoder = MjlogDecoder(modify=False)
    mjscore_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/mjscore")
    mjlog_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/mjlog")
    for mjscore in os.listdir(mjscore_dir):
        basename = os.path.splitext(mjscore)[0]
        mjscore_path = os.path.join(mjscore_dir, mjscore)
        mjlog_path = os.path.join(mjlog_dir, basename + ".mjlog")
        with open(mjscore_path, "r") as fs, open(mjlog_path, "r") as fl:
            lines = fl.readlines()
            assert len(lines) == 1
            mjlog = lines[0]
            mjprotos: List[mjproto.State] = mjlog_decoder.to_states(mjlog, store_cache=False)
            mjscores = fs.readlines()
            assert len(mjprotos) == len(mjscores)
            for mjproto, mjscore_original in zip(mjprotos, mjscores):
                mjscore_converted = mjproto_to_mjscore(mjproto)
                mjscore_converted_dict = json.loads(mjscore_converted)
                mjscore_original_dict = json.loads(mjscore_original)
                original_log = mjscore_original_dict["log"][0]  # logのみを比べる
                converted_log = mjscore_converted_dict["log"][0]
                for i in range(len(original_log[:-1])):  # 現状は局の結果は評価に入れない
                    assert (
                        original_log[i] == converted_log[i]
                    )  # TODO: replace with equality check function
