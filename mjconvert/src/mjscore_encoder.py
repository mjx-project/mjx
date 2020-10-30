import os
from typing import List
import copy
import json
import urllib.parse
from google.protobuf import json_format

import mj_pb2

#mjproto 形式の牌の表現をmjscore形式の表現に変える関数
def format_changer(protos :List[int])->List[int]:
    excepts = [16,52,88,"tumogiri"]#例外:赤５とツモぎり
    excepts_dict = {16:51,52:52,88:53,"tumogiri":60}
    scores = list(map(lambda x:((x//36)+1)*10 + ((x%36)//4)+1 if not x in excepts else excepts_dict[x],protos))#mjproto形式の表現ををmjscore形式に変換
    return scores
#mjscore形式の配牌をソートする関数。
def init_hand_sort(init_hand: List[int])->List[int]:
    # 11~19マンズ　21~29ピンズ　31~39ソウズ　#51~53赤マンピンソウ
    reds_score = [51, 52, 53]  # 赤
    init_key = [int(str(i)[::-1])+0.1 if i in reds_score else i for i in init_hand]  # ソートする辞書のキー。赤は文字を反転させて、同じ種類の牌の中で最後尾になるようにソートする値に0.1を足す。
    init_hand_dict = [[k, v] for k, v in zip(init_key, init_hand)]
    sorted_hand = sorted(init_hand_dict, key=lambda x: x[0])
    sorted_hand = [i[1] for i in sorted_hand]
    return sorted_hand

#mjproto 形式のeventを受け取り、あるプレイヤーの捨て牌をmjscore形式で出力する関数。
def discard_parser(events, abs_pos:int):
    discards = []
    for i in range(len(events)):
        event = events[i]
        if event.type==1  and event.who==abs_pos:#手出し
            discards.append(event.tile)
        elif event.type==2  and event.who==abs_pos:#ツモギリ
            discards.append("tumogiri")
        elif event.type==3 and event.who==abs_pos:#リーチ
            riichi = events[i+1].tile
            riich_index = discards.index(riichi)#riichiのeventの次に宣言牌が記録されているのでそのindexを記録
        else:
            None
    discards = format_changer(discards)#mjscore形式に変更
    try:
        discards[riich_index] = "r" + str(discards[riich_index])#リーチが行われていれば実行
    except:
        None
    return discards
def draw_parser(draws:List[int],events,abs_pos:int):
    """
    mjscoreでは引いた牌のリストにチーやポンなどのアクションが含まれている。。
    mjprotoの　drawsでは単に飛ばされていて、eventの方に情報がある。
    方策
    ①eventの方でアクション（ポン、チー、カン）と、直前の捨て牌を特定
    ②drawsの方で直前の捨て牌の直後にアクションを挿入。
    """
# ここを実装
def mjproto_to_mjscore(state: mj_pb2.State) -> str:
    #print(state.init_score.round)
    #print(state.private_infos.ABSOLUTE_POS_INIT_EAST.init_hand)
    # print(state.init_score.honba)
    #print(state.init_score.ten)
    #print(discard_parser(state.event_history.events,0))
    #print(len(state.private_infos[3].draws))
    print(init_hand_sort(format_changer(state.private_infos[0].init_hand)))
    round:int = state.init_score.round
    honba:int = state.init_score.honba
    riichi:int = state.init_score.riichi
    doras:List[int] = [i for i in state.doras]
    ura_doras:List[int] = [i for i in state.ura_doras]
    init_score:List[int] = [i for i in state.init_score.ten]

    d = {'title':[], 'name':[], 'rule':[],'log':[[[[round,honba,riichi],init_score,doras,ura_doras]]]}
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

