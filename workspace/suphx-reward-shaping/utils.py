import json
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
from google.protobuf import json_format

sys.path.append("../../")
sys.path.append("../../../")
import mjxproto

game_rewards = [90, 45, 0, -135]


def to_data(mjxprotp_dir: str, round_candidate=7, params=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    jsonが入っているディレクトリを引数としてjax.numpyのデータセットを作る.
    """
    features: List = []
    scores: List = []
    next_features = []
    for _json in os.listdir(mjxprotp_dir):
        _json = os.path.join(mjxprotp_dir, _json)
        assert ".json" in _json
        with open(_json, "r") as f:
            lines = f.readlines()
            _dicts = [json.loads(round) for round in lines]
            states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
            state = _select_one_round(states, round_candidate)
            if params:
                assert 0 <= round_candidate <= 6
                next_state = _select_one_round(states, round_candidate + 1)
                if not next_state:
                    continue
                next_features.append(to_feature(next_state, round_candidate))
            if not state:  # 該当する局がない場合飛ばす.
                continue
            feature: List = to_feature(state, round_candidate=round_candidate)
            features.append(feature)
            if params:
                continue
            else:
                scores.append(to_final_game_reward(states))
    features_array: jnp.ndarray = jnp.array(features)
    if params:
        assert next_features
        x = jnp.array(next_features)
        for i, param in enumerate(params.values()):
            x = jnp.dot(x, param)
            if i + 1 < len(params.values()):
                x = jax.nn.relu(x)
        scores_array = x
    else:
        scores_array: jnp.ndarray = jnp.array(scores)
    return features_array, scores_array


def _filter_round(states: List[mjxproto.State], candidate: int) -> List[int]:
    indices = []
    for idx, state in enumerate(states):
        if state.public_observation.init_score.round == candidate:
            indices.append(idx)
    return indices


def _select_one_round(
    states: List[mjxproto.State], candidate: Optional[int] = None
) -> Optional[mjxproto.State]:
    """
    データセットに本質的で無い相関が生まれることを防ぐために一半荘につき1ペアのみを使う.
    """
    if candidate:
        indices = _filter_round(states, candidate)
        if len(indices) == 0:  # 該当する局がない場合は飛ばす.
            return None
        idx = random.choice(indices)
        return states[idx]
    else:
        idx: int = random.randint(0, len(states) - 1)
        return states[idx]


def _calc_curr_pos(init_pos: int, round: int) -> int:
    pos = (-init_pos + round) % 4
    assert 0 <= pos <= 3
    return pos


def _calc_wind(init_pos: int, round: int) -> int:
    pos = (-init_pos + round) % 4
    if pos == 1:
        return 3
    if pos == 3:
        return 1
    return pos


def _to_one_hot(total_num: int, idx: int) -> List[int]:
    _l = [0] * total_num
    _l[idx] = 1
    return _l


def _clip_round(round: int, lim=7) -> int:
    """
    天鳳ではでは最長西4局まで行われるが何四局以降はサドンデスなので同一視.
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


def _remaining_oya(round: int):  # 局終了時の残りの親の数
    return [2 - (round // 4 + ((round % 4) >= i)) for i in range(4)]


def to_feature(
    state: mjxproto.State,
    is_round_one_hot=False,
    round_candidate: Optional[int] = None,
) -> List:
    """
    特徴量 = [4playerの点数, 起家の風:one-hot, 親:one-hot, 残りの親の数, 局, 本場, 詰み棒]
    """
    scores: List = [i / 100000 for i in state.public_observation.init_score.tens]
    honba: int = state.public_observation.init_score.honba
    tsumibo: int = state.public_observation.init_score.riichi
    round: int = _clip_round(state.public_observation.init_score.round)
    wind: List[int] = _to_one_hot(4, _calc_wind(0, round))  # 起家の風のみを入力
    oya: List[int] = _to_one_hot(4, _calc_curr_pos(0, round))
    remainning_oya = _remaining_oya(round)
    if is_round_one_hot:
        one_hot_round: List[int] = _to_one_hot(8, round)
        feature = (
            scores + wind + oya + remainning_oya + one_hot_round + [honba / 4, tsumibo / 4]
        )  # len(feature) = 26
    else:
        feature = (
            scores + wind + oya + remainning_oya + [round / 7, honba / 4, tsumibo / 4]
        )  # len(feature) = 19
    return feature


def to_final_game_reward(states: List[mjxproto.State]) -> List:
    """
    順位点. 起家から順番に. 4次元.
    """
    final_state = states[-1]
    final_scores = final_state.round_terminal.final_score.tens
    sorted_scores = sorted(final_scores, reverse=True)
    ranks = [sorted_scores.index(final_scores[i]) for i in range(4)]
    return [game_rewards[i] / 100 for i in ranks]


def _create_data_for_plot(score: int, round: int, is_round_one_hot, target: int) -> List:
    scores = [(100000 - score) / 300000] * 4
    scores[target] = score / 100000
    wind = _to_one_hot(4, _calc_wind(0, round))
    oya: List[int] = _to_one_hot(4, _calc_curr_pos(0, round))
    remainning_oya = _remaining_oya(round)
    if is_round_one_hot:
        rounds = [0] * 8
        rounds[round] = 1
        return scores + wind + oya + remainning_oya + rounds + [0, 0]
    else:
        return scores + wind + oya + remainning_oya + [round / 7, 0, 0]
