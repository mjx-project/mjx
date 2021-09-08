import json
import os
from typing import List

import mjxproto
from mjx.converter.mjlog_decoder import MjlogDecoder
from mjx.converter.mjscore_encoder import mjxproto_to_mjscore


def mjscore_log_equal(mjscore_original_dict, mjscore_converted_dict, proto) -> bool:
    original_log = mjscore_original_dict["log"][0]  # logのみを比べる
    converted_log = mjscore_converted_dict["log"][0]
    is_equal = True
    for i in range(len(original_log)):
        if original_log[i] != converted_log[i]:
            print(original_log[i], converted_log[i], proto.terminal.wins[0].ten_changes)
            is_equal = False
    return is_equal


def test_mjxproto_to_mjscore():
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
            mjxprotos: List[mjxproto.State] = mjlog_decoder.to_states(mjlog)
            mjscores = fs.readlines()
            assert len(mjxprotos) == len(mjscores)
            for proto, score_original in zip(mjxprotos, mjscores):
                score_converted = mjxproto_to_mjscore(proto)
                score_converted_dict = json.loads(score_converted)
                score_original_dict = json.loads(score_original)
                assert mjscore_log_equal(score_original_dict, score_converted_dict, proto)
