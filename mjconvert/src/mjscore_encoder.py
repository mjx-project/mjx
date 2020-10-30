import os
from typing import List
import copy
import json
import urllib.parse
from google.protobuf import json_format

import mj_pb2


def change_tile_fmt(tile_id: int) -> int:
    reds_in_mjproto = [16, 52, 88]
    reds_in_mjscore = [51, 52, 53]
    reds_dict = dict(zip(reds_in_mjproto, reds_in_mjscore))
    if tile_id in reds_in_mjproto:
        tile_in_mjscore = reds_dict[tile_id]
    else:
        tile_in_mjscore = ((tile_id // 36) + 1) * 10 + ((tile_id % 36) // 4) + 1
    return tile_in_mjscore


# mjproto形式の牌のリストを引数にとり、表現をmjscore形式の表現に変える関数
def change_tiles_fmt(tile_ids):
    scores = list(map(change_tile_fmt, tile_ids))  # mjproto形式の表現ををmjscore形式に変換
    return scores


# ["アクションの種類",晒した牌]の形式のリストを受け取ってmjscore形式に変更する関数
def change_action_format(action_list: List) -> str:
    # まだわかっていない。
    return "yet"


# mjscore形式の配牌をソートする関数。
def sort_init_hand(init_hand: List[int]) -> List[int]:
    # 11~19マンズ　21~29ピンズ　31~39ソウズ　#51~53赤マンピンソウ
    reds_score = [51, 52, 53]  # 赤
    init_key = [int(str(i)[::-1]) + 0.1 if i in reds_score else i for i in
                init_hand]  # ソートする辞書のキー。赤は文字を反転させて、同じ種類の牌の中で最後尾になるようにソートする値に0.1を足す。
    init_hand_dict = [[k, v] for k, v in zip(init_key, init_hand)]
    sorted_hand = sorted(init_hand_dict, key=lambda x: x[0])
    sorted_hand = [i[1] for i in sorted_hand]
    return sorted_hand


# mjproto形式のeventを受け取り、あるプレイヤーの捨て牌をmjscore形式で出力する関数。
def parse_discards(events, abs_pos: int):
    discards = []
    is_reach = False  # リーチの有無
    for i, event in enumerate(events):
        if event.type == mj_pb2.EVENT_TYPE_DISCARD_FROM_HAND and event.who == abs_pos:  # 手出し
            discards.append(change_tile_fmt(event.tile))
        elif event.type == mj_pb2.EVENT_TYPE_DISCARD_DRAWN_TILE and event.who == abs_pos:  # ツモギリ
            discards.append(60)
        elif event.type == mj_pb2.EVENT_TYPE_RIICHI and event.who == abs_pos:  # リーチ
            is_reach = True
            riichi_tile = change_tile_fmt(events[i + 1].tile)  # riichiの次のeventに宣言牌が記録されているのでそのtileを取得して後にindexを取得変更
    riichi_index = discards.index(riichi_tile)
    if is_reach:
        discards[riichi_index] = "r" + str(discards[riichi_index])  # リーチ宣言牌の形式変更
    return discards


# プレイヤーごとにmjscore形式のdrawsを返す。
def parse_draws(draws, events, abs_pos):
    """
    - mjscoreでは引いた牌のリストにチーやポンなどのアクションが含まれている
    - mjprotoの　drawsでは単に飛ばされていて、eventの方に情報がある

    方針
    1. チーポンも含めたdiscardsを作成
    2. drawsの方で直前の捨て牌の直後にアクションを挿入
    """
    draws = change_tiles_fmt(draws)
    discards = []
    for event in events:
        if event.type == mj_pb2.EVENT_TYPE_DISCARD_FROM_HAND and event.who == abs_pos:  # 手出し
            discards.append(event.tile)
        elif event.type == mj_pb2.EVENT_TYPE_DISCARD_DRAWN_TILE and event.who == abs_pos:  # ツモギリ
            discards.append(60)
        elif event.type == mj_pb2.EVENT_TYPE_CHI and event.who == abs_pos:  # チー
            discards.append(["chi", event.open])
        elif event.type == mj_pb2.EVENT_TYPE_PON and event.who == abs_pos:  # ポン
            discards.append(["pon", event.open])
    for i in range(len(discards)):
        if type(discards[i]) == list:
            draws.insert(i, change_action_format(discards[i]))  # チーやポンのアクションをdrawsに挿入

    return draws


# ここを実装
def mjproto_to_mjscore(state: mj_pb2.State) -> str:
    # print(state.init_score.round)
    # print(state.private_infos.ABSOLUTE_POS_INIT_EAST.init_hand)
    # print(state.init_score.honba)
    # print(state.init_score.ten)
    # print(discard_parser(state.event_history.events,0))
    # print(len(state.private_infos[3].draws))
    # print(sort_init_hand(change_tiles_fmt(state.private_infos[0].init_hand)))
    print(parse_discards(state.event_history.events, 1))
    round: int = state.init_score.round
    honba: int = state.init_score.honba
    riichi: int = state.init_score.riichi
    doras: List[int] = [i for i in state.doras]
    ura_doras: List[int] = [i for i in state.ura_doras]
    init_score: List[int] = [i for i in state.init_score.ten]

    d = {'title': [], 'name': [], 'rule': [], 'log': [[[[round, honba, riichi], init_score, doras, ura_doras]]]}
    return json.dumps(d)


if __name__ == '__main__':
    # 東1局0本場の mjproto
    path_to_mjproto_example = "../..//test/resources/json/first-example.json"
    with open(path_to_mjproto_example, 'r') as f:
        line = f.readline()
    d = json.loads(line)
    state: mj_pb2.State = json_format.ParseDict(d, mj_pb2.State())

    # 東1局0本場の mjscore
    path_to_mjscore_example = "../../test/resources/mjscore/first-example.json"
    with open(path_to_mjscore_example, 'r') as f:
        line = f.readline()
    mjscore_expected_dict = json.loads(line)

    # 実装を使って変換
    mjscore_str = mjproto_to_mjscore(state)
    mjscore_dict = json.loads(mjscore_str)

    # 比較
    print(mjscore_expected_dict)
    print(mjscore_dict)
    print(mjscore_expected_dict == mjscore_dict)
