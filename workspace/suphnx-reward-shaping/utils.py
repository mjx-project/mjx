import json
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from google.protobuf import json_format

sys.path.append("../../")
import mjxproto


def to_dataset(mjxprotp_dir: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    jsonが入っているディレクトリを引数としてjax.numpyのデータセットを作る.
    """
    features: List = []
    scores: List = []
    for _json in os.listdir(mjxprotp_dir):
        _json = os.path.join(mjxprotp_dir, _json)
        assert ".json" in _json
        with open(_json, "r") as f:
            lines = f.readlines()
            _dicts = [json.loads(round) for round in lines]
            states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
            target: int = random.randint(0, 3)
            features.append(to_feature(states, target))
            scores.append(to_final_scores(states, target))
    features_array: jnp.ndarray = jnp.array(features)
    scores_array: jnp.ndarray = jnp.array(scores)
    return features_array, scores_array


def _select_one_round(states: List[mjxproto.State]) -> mjxproto.State:
    """
    データセットに本質的で無い相関が生まれることを防ぐために一半荘につき1ペアのみを使う.
    """
    idx: int = random.randint(0, len(states) - 1)
    return states[idx]


def _calc_curr_pos(init_pos: int, round: int) -> int:
    return init_pos + round % 4


def to_feature(states: List[mjxproto.State], target) -> List[int]:
    """
    特徴量 = [終了時の点数, 自風, 親, 局, 本場, 詰み棒]
    """
    state = _select_one_round(states)
    ten: int = state.round_terminal.final_score.tens[target]
    honba: int = state.round_terminal.final_score.honba
    tsumibo: int = state.round_terminal.final_score.riichi
    round: int = state.round_terminal.final_score.round
    wind: int = _calc_curr_pos(target, round)
    oya: int = _calc_curr_pos(0, round)
    return [ten, honba, tsumibo, round, wind, oya]


def to_final_scores(states: List[mjxproto.State], target) -> List[int]:
    final_state = states[-1]
    final_score = final_state.round_terminal.final_score.tens[target]
    return [final_score]
