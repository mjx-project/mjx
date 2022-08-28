import json
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from google.protobuf import json_format

sys.path.append("../../")
import mjxproto


def to_dataset(mjxprotp_dir: str):
    features: List[List[int]] = []
    scores: List[int] = []
    for _json in os.listdir(mjxprotp_dir):
        _json = os.path.join(mjxprotp_dir, _json)
        assert ".json" in _json
        with open(_json, "r") as f:
            lines = f.readlines()
            target: int = random.randint(0, 3)
            features.append(to_feature(lines, target))
            scores.append(to_final_scores(lines, target))
    features = jnp.array(features)
    scores = jnp.array(scores)
    return features, scores


def to_feature(game_str: List[str], target) -> np.ndarray:
    """
    特徴量 = [終了時の点数, 自風, 親, 局, 本場, 詰み棒]
    """
    _dicts = [json.loads(round) for round in game_str]
    states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
    _s = select_one_round(states)
    feature: List[int] = extract_feature(_s, target)
    return feature


def select_one_round(states: List[mjxproto.State]) -> mjxproto.State:
    idx: int = random.randint(0, len(states) - 1)
    return states[idx]


def extract_feature(state: mjxproto.State, target) -> List[int]:
    ten: int = state.round_terminal.final_score.tens[target]
    honba: int = state.round_terminal.final_score.honba
    tsumibo: int = state.round_terminal.final_score.riichi
    round: int = state.round_terminal.final_score.round
    wind: int = calc_curr_pos(target, round)
    oya: int = calc_curr_pos(0, round)
    return [ten, honba, tsumibo, round, wind, oya]


def calc_curr_pos(init_pos: int, round: int) -> int:
    return init_pos + round % 4


def to_final_scores(game_str: List[str], target) -> List[int]:
    _dicts = [json.loads(round) for round in game_str]
    states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
    final_state = states[-1]
    final_score = final_state.round_terminal.final_score.tens[target]
    return [final_score]
