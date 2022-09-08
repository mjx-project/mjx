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
sys.path.append("../../../")
import mjxproto

game_rewards = [90, 45, 0, -135]


def to_data(
    mjxprotp_dir: str, round_candidates: Optional[List[int]] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            features.append(to_feature(states, target, round_candidates=round_candidates))
            scores.append(to_final_game_reward(states, target))
    features_array: jnp.ndarray = jnp.array(features)
    scores_array: jnp.ndarray = jnp.array(scores)
    return features_array, scores_array


def _select_one_round(
    states: List[mjxproto.State], candidates: Optional[List[int]] = None
) -> mjxproto.State:
    """
    データセットに本質的で無い相関が生まれることを防ぐために一半荘につき1ペアのみを使う.
    """
    if candidates:
        if min(candidates) > len(states) - 1:  # 候補のと対応する局がない場合, 一番近いものを返す.
            return states[len(states) - 1]
        idx = random.choice(candidates)
        return states[idx]
    else:
        idx: int = random.randint(0, len(states) - 1)
        return states[idx]


def _calc_curr_pos(init_pos: int, round: int) -> int:
    pos = (init_pos + round) % 4
    assert 0 <= pos <= 3
    return pos


def _to_one_hot(total_num: int, idx: int) -> List[int]:
    _l = [0] * total_num
    _l[idx] = 1
    return _l


def _clip_round(round: int, lim=7) -> int:
    """
    天鳳ではてんほうでは最長西4局まで行われるが何四局以降はサドンデスなので同一視.
    """
    if round < 7:
        return round
    else:
        return 7


def _preprocess_scores(scores, target: int) -> List:
    """
    局終了時の点数を100000で割って自家, 下家, 対面, 上家の順に並び替える.
    """
    _self: int = scores[target] / 100000
    _left: int = scores[target - 1] / 100000
    _front: int = scores[target - 2] / 100000
    _right: int = scores[target - 3] / 100000
    return [_self, _left, _front, _right]


def to_feature(
    states: List[mjxproto.State],
    target,
    is_round_one_hot=False,
    round_candidates: Optional[List[int]] = None,
) -> List:
    """
    特徴量 = [4playerの点数, 自風:one-hot, 親:one-hot, 局, 本場, 詰み棒]
    """
    state = _select_one_round(states, candidates=round_candidates)
    scores: List = _preprocess_scores(state.round_terminal.final_score.tens, target)
    honba: int = state.round_terminal.final_score.honba
    tsumibo: int = state.round_terminal.final_score.riichi
    round: int = _clip_round(state.round_terminal.final_score.round)
    wind: List[int] = _to_one_hot(4, _calc_curr_pos(target, round))
    oya: List[int] = _to_one_hot(4, _calc_curr_pos(0, round))
    if is_round_one_hot:
        one_hot_round: List[int] = _to_one_hot(8, round)
        feature = (
            scores + wind + oya + one_hot_round + [honba / 4, tsumibo / 4]
        )  # len(feature) = 22
    else:
        feature = scores + wind + oya + [round / 7, honba / 4, tsumibo / 4]  # len(feature) = 15
    return feature


def to_final_game_reward(states: List[mjxproto.State], target) -> List:
    """
    順位点.
    """
    final_state = states[-1]
    final_scores = final_state.round_terminal.final_score.tens
    target_score = final_scores[target]
    sorted_scores = sorted(final_scores, reverse=True)
    rank = sorted_scores.index(target_score)
    return [game_rewards[rank] / 100]
