import json
import os
import sys
from concurrent.futures import process

import numpy as np
from google.protobuf import json_format

sys.path.append("../../../")
import mjxproto

sys.path.append("../")
from train_helper import initializa_params
from utils import (
    _calc_wind,
    _preprocess_score,
    _preprocess_score_inv,
    to_data,
    to_final_game_reward,
)

mjxprotp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def test_preprocess():
    """
    game rewardの前処理のテスト
    """
    assert _preprocess_score(90) == 1
    assert _preprocess_score(-135) == 0


def test_preprocess_inv():
    """
    activation functionをlogistic関数にして, 元のスコアにうまく変換できるか
    """
    assert _preprocess_score_inv(_preprocess_score(90)) == 90
    assert -0.0001 <= _preprocess_score_inv(_preprocess_score(0)) <= 0.0001
    assert 44.99999 <= _preprocess_score_inv(_preprocess_score(45)) <= 45.0001
    assert -135.00005 <= _preprocess_score_inv(_preprocess_score(-135)) <= -134.99999


def test_calc_wind():
    """
    風の計算のテスト
    """
    assert _calc_wind(1, 0) == 1
    assert _calc_wind(1, 3) == 2


def test_to_final_game_reward():
    """
    game reward計算のテスト.
    """
    for i in range(4):
        scores = [
            [1.0, 0.6, 0.0, 0.8],
            [0.6, 0, 0.8, 1.0],
            [1.0, 0, 0.6, 0.8],
            [0.0, 1.0, 0.6, 0.8],
        ]
        _dir = os.path.join(mjxprotp_dir, os.listdir(mjxprotp_dir)[i])
        with open(_dir, "r") as f:
            lines = f.readlines()
            _dicts = [json.loads(round) for round in lines]
            states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
            print(states[-1].round_terminal.final_score.tens)
            assert to_final_game_reward(states) == scores[i]


def test_to_data():
    """
    データ生成テスト
    """
    num_resources = len(os.listdir(mjxprotp_dir))
    features, target, scores = to_data(mjxprotp_dir, round=7)
    assert features.shape == (num_resources, 19)
    assert scores.shape == (num_resources, 4)
