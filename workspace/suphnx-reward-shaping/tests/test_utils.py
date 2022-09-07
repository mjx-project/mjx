import json
import os
import sys

from google.protobuf import json_format

sys.path.append("../../../")
import mjxproto

sys.path.append("../")
from utils import _preprocess_scores, to_data, to_final_game_reward

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_preprocess():
    scores = [0, 100000, 200000, 300000]
    print(_preprocess_scores(scores, 1))
    assert _preprocess_scores(scores, 0) == [0, 3, 2, 1]
    assert _preprocess_scores(scores, 1) == [1, 0, 3, 2]
    assert _preprocess_scores(scores, 2) == [2, 1, 0, 3]
    assert _preprocess_scores(scores, 3) == [3, 2, 1, 0]


def test_to_data():
    num_resources = len(os.listdir(mjxprotp_dir))
    features, scores = to_data(mjxprotp_dir)
    print(features)
    assert features.shape == (num_resources, 15)
    assert scores.shape == (num_resources, 1)
