import os
from typing import List

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
                assert (
                    mjscore_original == mjscore_converted
                )  # TODO: replace with equality check function
