import json
import os
import sys

from google.protobuf import json_format

sys.path.append("../../../")
import mjxproto

sys.path.append("../")
from train_helper import initializa_params
from utils import _calc_wind, _preprocess_scores, to_data, to_final_game_reward

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_preprocess():
    scores = [0, 100000, 200000, 300000]
    print(_preprocess_scores(scores, 1))
    assert _preprocess_scores(scores, 0) == [0, 3, 2, 1]
    assert _preprocess_scores(scores, 1) == [1, 0, 3, 2]
    assert _preprocess_scores(scores, 2) == [2, 1, 0, 3]
    assert _preprocess_scores(scores, 3) == [3, 2, 1, 0]


def test_calc_wind():
    assert _calc_wind(1, 0) == 1
    assert _calc_wind(1, 3) == 2


def test_to_final_game_reward():
    for i in range(4):
        scores = [
            [0.9, 0.0, -1.35, 0.45],
            [0.0, -1.35, 0.45, 0.9],
            [0.9, -1.35, 0.0, 0.45],
            [-1.35, 0.9, 0.0, 0.45],
        ]
        _dir = os.path.join(mjxprotp_dir, os.listdir(mjxprotp_dir)[i])
        with open(_dir, "r") as f:
            lines = f.readlines()
            _dicts = [json.loads(round) for round in lines]
            states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
            print(states[-1].round_terminal.final_score.tens)
            assert to_final_game_reward(states) == scores[i]


def test_to_data():
    num_resources = len(os.listdir(mjxprotp_dir))
    features, scores = to_data(mjxprotp_dir)
    assert features.shape == (num_resources, 19)
    assert scores.shape == (num_resources, 4)


if __name__ == "__main__":
    test_to_final_game_reward()
