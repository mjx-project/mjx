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


def to_data(
    mjxprotp_dir: str,
    round=None,
    params=None,
    use_logistic=False,
    use_clip=True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    jsonが入っているディレクトリを引数としてjax.numpyのデータセットを作る.
    """
    features: List = []
    targets: List = []
    scores: List = []
    next_features = []
    for _json in os.listdir(mjxprotp_dir):
        _json = os.path.join(mjxprotp_dir, _json)
        with open(_json, errors="ignore") as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            _dicts = [json.loads(round) for round in lines]
            states = [json_format.ParseDict(d, mjxproto.State()) for d in _dicts]
            state = _select_one_round(states, round=round)
            if params:
                assert (round != None) and (0 <= round <= 6)
                next_state = _select_one_round(states, round=round + 1)
                if not next_state:
                    continue
                next_features.append(to_feature(next_state))
            if not state:  # 該当する局がない場合飛ばす.
                continue
            feature: List = to_feature(state)
            features.append(feature)
            targets.append(to_final_game_reward(states))
            scores.append(list(map(_preprocess_score_inv, to_final_game_reward(states))))
    features_array: jnp.ndarray = jnp.array(features)
    scores_array: jnp.ndarray = jnp.array(scores)
    if params:
        x = jnp.array(next_features)
        for i, param in enumerate(params.values()):
            x = jnp.dot(x, param)
            if i + 1 < len(params.values()):
                x = jax.nn.relu(x)
        if use_logistic:
            targets_array: jnp.ndarray = jnp.exp(x) / (1 + jnp.exp(x))
        elif use_clip:
            targets_array: jnp.ndarray = jnp.clip(x, a_min=0, a_max=1)
        else:
            targets_array: jnp.ndarray = x

    else:
        targets_array: jnp.ndarray = jnp.array(targets)
    return (features_array, targets_array, scores_array)


def _filter_round(states: List[mjxproto.State], round: int) -> List[int]:
    indices = []
    for idx, state in enumerate(states):
        if state.public_observation.init_score.round == round:
            indices.append(idx)
    return indices


def _select_one_round(
    states: List[mjxproto.State], round: Optional[int] = None
) -> Optional[mjxproto.State]:
    """
    データセットに本質的で無い相関が生まれることを防ぐために一半荘につき1ペアのみを使う.
    """
    if round != None:
        indices = _filter_round(states, round)
        if len(indices) == 0:  # 該当する局がない場合は飛ばす.
            return None
        idx = random.choice(indices)
        return states[idx]
    else:
        state: mjxproto.State = random.choice(states)
        return state


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


def _remaining_oya(round: int):  # 局終了時の残りの親の数
    return [2 - (round // 4 + ((round % 4) >= i)) for i in range(4)]


def to_feature(
    state: mjxproto.State,
    is_round_one_hot=False,
) -> List:
    """
    特徴量 = [4playerの点数, 起家の風:one-hot, 親:one-hot, 残りの親の数, 局, 本場, 詰み棒]
    """
    scores: List = [i / 100000 for i in state.public_observation.init_score.tens]  # 100000で割る.
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


def _preprocess_score(score):
    """
    game rewardを0~1に正規化
    """
    return (score + 135) / 225


def _preprocess_score_inv(processed_score):
    """
    変換したtargetを用いて学習したNNの出力を元々のscoreの範囲に変換
    """
    return 225 * (processed_score) - 135


def to_final_game_reward(states: List[mjxproto.State]) -> List:
    """
    順位点. 起家から順番に. 4次元.
    """
    final_state = states[-1]
    final_scores = final_state.round_terminal.final_score.tens
    sorted_scores = sorted(final_scores, reverse=True)
    ranks = [sorted_scores.index(final_scores[i]) for i in range(4)]
    return [_preprocess_score(game_rewards[i]) for i in ranks]


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
