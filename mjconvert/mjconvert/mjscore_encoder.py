from __future__ import annotations  # postpone type hint evaluation or doctest fails

import json
from typing import Dict, List

import mjproto
from mjconvert import open_converter


def _change_tile_fmt(tile_id: int) -> int:
    reds_in_mjproto = [16, 52, 88]
    reds_in_mjscore = [51, 52, 53]
    reds_dict = dict(zip(reds_in_mjproto, reds_in_mjscore))
    if tile_id in reds_in_mjproto:
        tile_in_mjscore = reds_dict[tile_id]
    else:
        tile_in_mjscore = ((tile_id // 36) + 1) * 10 + ((tile_id % 36) // 4) + 1
    return tile_in_mjscore


# mjproto形式の牌のリストを引数にとり、表現をmjscore形式の表現に変える関数
def _change_tiles_fmt(tile_ids):
    scores = list(map(_change_tile_fmt, tile_ids))  # mjproto形式の表現ををmjscore形式に変換
    return scores


# openの情報を受け取ってmjscore形式に変更する関数
def _change_action_format(bits: int) -> str:  # TODO カン
    event_type = open_converter.open_event_type(bits)
    open_from = open_converter.open_from(bits)
    stolen_tile = open_converter.change_open_tile_fmt(open_converter.open_stolen_tile_type(bits))
    open_tiles = open_converter.change_open_tiles_fmt(open_converter.open_tile_types(bits))
    open_tiles.remove(stolen_tile)
    if event_type == mjproto.EVENT_TYPE_CHI:  # チー
        return "c" + str(stolen_tile) + str(open_tiles[0]) + str(open_tiles[1])
    elif event_type == mjproto.EVENT_TYPE_PON:  # ポン
        if open_from == mjproto.RELATIVE_POS_LEFT:
            return "p" + str(stolen_tile) + str(open_tiles[0]) + str(open_tiles[1])
        elif open_from == mjproto.RELATIVE_POS_MID:
            return str(stolen_tile) + "p" + str(open_tiles[0]) + str(open_tiles[1])
        else:
            return str(stolen_tile) + str(open_tiles[0]) + "p" + str(open_tiles[1])
    elif event_type == mjproto.EVENT_TYPE_KAN_ADDED:  # 加槓
        if open_from == mjproto.RELATIVE_POS_LEFT:
            return "k" + str(stolen_tile) + str(open_tiles[0]) + str(open_tiles[1]) + str(open_tiles[2])
        elif open_from == mjproto.RELATIVE_POS_MID:
            return str(open_tiles[0]) + "k" + str(stolen_tile) + str(open_tiles[1]) + str(open_tiles[2])
        else:
            return str(open_tiles[0]) + str(open_tiles[1]) + "k" + str(stolen_tile) + str(open_tiles[2])
    elif event_type == mjproto.EVENT_TYPE_KAN_CLOSED:  # 暗槓
        return str(stolen_tile) + str(stolen_tile) + str(stolen_tile) + "a" + str(stolen_tile)
    else:  # 明槓
        if open_from == mjproto.RELATIVE_POS_LEFT:
            return "m" + str(stolen_tile) + str(open_tiles[0]) + str(open_tiles[1]) + str(open_tiles[2])
        elif open_from == mjproto.RELATIVE_POS_MID:
            return str(stolen_tile) + "m" + str(open_tiles[0]) + str(open_tiles[1]) + str(open_tiles[2])
        else:
            return str(stolen_tile) + str(open_tiles[0]) + "m" + str(open_tiles[1]) + str(open_tiles[2])


# mjscore形式の配牌をソートする関数。
def sort_init_hand(init_hand: List) -> List:
    # 11~19マンズ　21~29ピンズ　31~39ソウズ　#51~53赤マンピンソウ
    reds_score = [51, 52, 53]  # 赤
    init_key = [
        int(str(i)[::-1]) + 0.1 if i in reds_score else i for i in init_hand
    ]  # ソートする辞書のキー。赤は文字を反転させて、同じ種類の牌の中で最後尾になるようにソートする値に0.1を足す。
    init_hand_dict = [[k, v] for k, v in zip(init_key, init_hand)]
    sorted_hand = sorted(init_hand_dict, key=lambda x: x[0])
    sorted_hand = [i[1] for i in sorted_hand]
    return sorted_hand


def _change_tumogiri_riich_fmt(tile):  # ツモギリリーチ専用の番号90を60ツモぎりの番号60に直す
    if tile == 90:
        return 60
    return tile


# mjproto形式のeventを受け取り、あるプレイヤーの捨て牌をmjscore形式で出力する関数。
def parse_discards(events, abs_pos: int):
    discards: List[object] = []
    for i, event in enumerate(events):
        if event.type == mjproto.EVENT_TYPE_DISCARD_FROM_HAND and event.who == abs_pos:  # 手出し
            if events[i - 1].type == mjproto.EVENT_TYPE_RIICHI:  # 一つ前のeventがriichiかどうか
                discards.append("r" + str(_change_tile_fmt(event.tile)))
            else:
                discards.append(_change_tile_fmt(event.tile))
        elif event.type == mjproto.EVENT_TYPE_DISCARD_DRAWN_TILE and event.who == abs_pos:  # ツモギリ
            if events[i - 1].type == mjproto.EVENT_TYPE_RIICHI:  # 一つ前のeventがriichiかどうか
                discards.append("r60")
            else:
                discards.append(60)
        elif event.type == mjproto.EVENT_TYPE_KAN_CLOSED and event.who == abs_pos:
            discards.append(_change_action_format(event.open))
        elif event.type == mjproto.EVENT_TYPE_KAN_ADDED and event.who == abs_pos:
            discards.append(_change_action_format(event.open))
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
    draws = _change_tiles_fmt(draws)
    discards = []
    actions = []  #
    for i, event in enumerate(events):
        if event.type == mjproto.EVENT_TYPE_DISCARD_FROM_HAND and event.who == abs_pos:  # 手出し
            discards.append(event.tile)
        elif event.type == mjproto.EVENT_TYPE_DISCARD_DRAWN_TILE and event.who == abs_pos:  # ツモギリ
            discards.append(60)
        elif event.type == mjproto.EVENT_TYPE_CHI and event.who == abs_pos:  # チー
            discards.append(event.open)
            actions.append(event.open)
        elif event.type == mjproto.EVENT_TYPE_PON and event.who == abs_pos:  # ポン
            discards.append(event.open)
            actions.append(event.open)
        elif event.type == mjproto.EVENT_TYPE_KAN_OPENED and event.who == abs_pos:  # 明槓
            discards.append(event.open)
            actions.append(event.open)
    for i, action in enumerate(actions):
        action_index = discards.index(action) - i  # 捨て牌でのactionのindex同じ順にdrawにアクションを挿入すれば良い
        draws.insert(action_index, _change_action_format(action))
    return draws


yaku_list_tumo = [
    "門前清自摸和(1飜)",
    "立直(1飜)",
    "一発(1飜)",
    "槍槓(1飜)",
    "嶺上開花(1飜)",
    "海底摸月(1飜)",
    "河底撈魚(1飜)",
    "平和(1飜)",
    "断幺九(1飜)",
    "一盃口(1飜)",
    "自風 東(1飜)",
    "自風 南(1飜)",
    "自風 西(1飜)",
    "自風 北(1飜)",
    "場風 東(1飜)",
    "場風 南(1飜)",
    "場風 西(1飜)",
    "場風 北(1飜)",
    "役牌 白(1飜)",
    "役牌 發(1飜)",
    "役牌 中(1飜)",
    "両立直(2飜)",
    "七対子(2飜)",
    "混全帯幺九(2飜)",
    "一気通貫(2飜)",
    "三色同順(2飜)",
    "三色同刻(2飜)",
    "三槓子(2飜)",
    "対々和(2飜)",
    "三暗刻(2飜)",
    "小三元(2飜)",
    "混老頭(2飜)",
    "二盃口(3飜)",
    "純全帯幺九(3飜)",
    "混一色(3飜)",
    "清一色(6飜)",
    "人和",  # 天鳳は人和なし
    "天和(役満)",
    "地和(役満)",
    "大三元(役満)",
    "四暗刻(役満)",
    "四暗刻単騎(役満)",
    "字一色(役満)",
    "緑一色(役満)",
    "清老頭(役満)",
    "九蓮宝燈(役満)",
    "純正九蓮宝燈(役満)",
    "国士無双(役満)",
    "国士無双１３面(役満)",
    "大四喜(役満)",
    "小四喜(役満)",
    "四槓子(役満)",
    "ドラ",
    "裏ドラ",
    "赤ドラ",
]

yaku_list_ron = [
    "門前清自摸和(1飜)",
    "立直(1飜)",
    "一発(1飜)",
    "槍槓(1飜)",
    "嶺上開花(1飜)",
    "海底摸月(1飜)",
    "河底撈魚(1飜)",
    "平和(1飜)",
    "断幺九(1飜)",
    "一盃口(1飜)",
    "自風 東(1飜)",
    "自風 南(1飜)",
    "自風 西(1飜)",
    "自風 北(1飜)",
    "場風 東(1飜)",
    "場風 南(1飜)",
    "場風 西(1飜)",
    "場風 北(1飜)",
    "役牌 白(1飜)",
    "役牌 發(1飜)",
    "役牌 中(1飜)",
    "両立直(2飜)",
    "七対子(2飜)",
    "混全帯幺九(1飜)",
    "一気通貫(1飜)",
    "三色同順(1飜)",
    "三色同刻(2飜)",
    "三槓子(2飜)",
    "対々和(2飜)",
    "三暗刻(2飜)",
    "小三元(2飜)",
    "混老頭(2飜)",
    "二盃口(3飜)",
    "純全帯幺九(2飜)",
    "混一色(2飜)",
    "清一色(5飜)",
    "人和",  # 天鳳は人和なし
    "天和(役満)",
    "地和(役満)",
    "大三元(役満)",
    "四暗刻(役満)",
    "四暗刻単騎(役満)",
    "字一色(役満)",
    "緑一色(役満)",
    "清老頭(役満)",
    "九蓮宝燈(役満)",
    "純正九蓮宝燈(役満)",
    "国士無双(役満)",
    "国士無双１３面(役満)",
    "大四喜(役満)",
    "小四喜(役満)",
    "四槓子(役満)",
    "ドラ",
    "裏ドラ",
    "赤ドラ",
]
yaku_list_keys = [i for i in range(55)]
yaku_dict_tumo = {k: v for k, v in zip(yaku_list_keys, yaku_list_tumo)}
yaku_dict_ron = {k: v for k, v in zip(yaku_list_keys, yaku_list_ron)}

non_dealer_tsumo_dict = {
    1100: "300-500",
    1500: "400-700",
    1600: "800-1600",
    2000: "500-1000",
    2400: "600-1200",
    2700: "700-1300",
    3100: "800-1500",
    3200: "800-1600",
    3600: "900-1800",
    4000: "1000-2000",
    4700: "1200-2300",
    5200: "1300-2600",
    5900: "1500-2900",
    6400: "1600-3200",
    7200: "1800-3600",
    7900: "2000-3900",
    8000: "2000-4000",
    12000: "3000-6000",
    16000: "4000-8000",
    24000: "6000-12000",
    32000: "8000-16000",
}

dealer_point_dict = {12000: "満貫", 18000: "跳満", 24000: "倍満", 36000: "三倍満", 48000: "役満"}
no_dealer_point_dict = {8000: "満貫", 12000: "跳満", 16000: "倍満", 24000: "三倍満", 32000: "役満"}


def _fan_fu(who, fans: List[int], fu: int, ten, round: int) -> str:
    """
    >>> _fan_fu(0, [3, 1], 40, 12000, 0)
    '満貫'
    >>> _fan_fu(1, [2, 1], 40, 5200, 0)
    '40符3飜'
    """
    fan = sum(fans)
    if _is_oya(who, round):  # 親かどうかを判定
        if ten < 12000:
            return str(fu) + "符" + str(fan) + "飜"
        else:
            return dealer_point_dict[ten]
    else:
        if int(ten) < 8000:
            return str(fu) + "符" + str(fan) + "飜"
        else:
            return no_dealer_point_dict[ten]


def _is_oya(who: int, round: int) -> bool:  # 和了者が親かどうか判定する。
    """
    >>> _is_oya(0, 3)
    False
    >>> _is_oya(0, 4)
    True
    """
    if round % 4 == who:
        return True
    else:
        return False


def _winner_point(who: int, from_who: int, fans: List[int], fu: int, ten: int, round: int) -> str:
    """
    >>> _winner_point(0, 0, [3, 0], 30, 6000, 0)
    '30符3飜2000点∀'
    >>> _winner_point(2, 0, [5, 0], 40, 8000, 0)
    '満貫8000点'
    """
    is_tsumo = who == from_who  # ツモあがりかどうかを判定
    if is_tsumo:
        if _is_oya(who, round):  # 親かどうか
            return _fan_fu(who, fans, fu, ten, round) + str(int(ten / 3)) + "点∀"
        else:
            return _fan_fu(who, fans, fu, ten, round) + non_dealer_tsumo_dict[ten] + "点"
    else:
        if who == mjproto.ABSOLUTE_POS_INIT_EAST:
            return _fan_fu(who, fans, fu, ten, round) + str(ten) + "点"
        else:
            return _fan_fu(who, fans, fu, ten, round) + str(ten) + "点"


def _check_uradoras(fans: List[int], yakus: List[int]) -> List[int]:  # リーチがかかるとprotoではyakus
    # に強制的にウラドラの情報が入るが、乗っているかどうかを確認する必要がある
    """
    >>> _check_uradoras([1, 1, 1, 0], [1, 0, 7, 53])
    [1, 0, 7]
    >>> _check_uradoras([1, 1, 2, 1, 2, 0], [1, 0, 29, 8, 54, 53])
    [1, 0, 29, 8, 54]
    """
    if sum(fans) < len(yakus):
        return [i for i in yakus if i != 53]
    elif fans[-1] == 0:  # 裏ドラは必ずfansの末尾に表示されるので0かどうかで判定がつく。
        return [i for i in yakus if i != 53]
    else:
        return yakus


def _correspond_yakus(yaku_dict, yakus: List[int], fans: List[int]):
    """
    >>> _correspond_yakus(yaku_dict_tumo, [0, 52], [1, 2])
    ['門前清自摸和(1飜)', 'ドラ(2飜)']
    """
    doras = [52, 53, 54]
    yakus_in_japanese = []
    for i in yakus:
        if i not in doras:
            yakus_in_japanese.append(yaku_dict[i])
    for i in doras:  # ドラの枚数はfansの対応するインデックスの情報からわかる。
        if i in yakus:
            d_idx = yakus.index(i)
            yakus_in_japanese.append(
                yaku_dict[i] + "({}飜)".format(str(fans[d_idx]))
            )  # ドラは複数ある場合はまとめてドラ(3飜)の様に表記
    return yakus_in_japanese


def _winner_yakus(yakus: List[int], fans: List[int]) -> List[str]:
    """
    >>> _winner_yakus([0, 1, 23], [1, 1, 2])
    ['門前清自摸和(1飜)', '立直(1飜)', '混全帯幺九(2飜)']
    >>> _winner_yakus([23], [1])
    ['混全帯幺九(1飜)']
    """
    if 0 in yakus:  # ツモの有無によって役の飜数がかわる。
        return _correspond_yakus(yaku_dict_tumo, yakus, fans)
    else:
        return _correspond_yakus(yaku_dict_ron, yakus, fans)


def parse_terminal(state: mjproto.State):
    if len(state.terminal.wins) == 0:  # あがった人がいない場合,# state.terminal.winsの長さは0
        ten_changes = [i for i in state.terminal.no_winner.ten_changes]
        return ["流局", ten_changes]
    else:
        round = state.init_score.round
        who = state.terminal.wins[0].who
        from_who = state.terminal.wins[0].from_who
        ten_changes = [i for i in state.terminal.wins[0].ten_changes]
        fans = [i for i in state.terminal.wins[0].fans]  # [役での飜数, ドラの数]
        yakus = _check_uradoras(fans, [i for i in state.terminal.wins[0].yakus])
        fu = state.terminal.wins[0].fu
        ten = state.terminal.wins[0].ten
        """
        情報
        fnas:[役での飜数, ドラでの飜数]
        yakus: [役とドラの種類]
        ten: 純粋に上がり点が表示される。ツモ上がりの際の対応が必要
        """
        yaku_point_infos = [who, from_who, who, _winner_point(who, from_who, fans, fu, ten, round)]
        yaku_point_infos.extend(_winner_yakus(yakus, fans))
        return [
            "和了",
            ten_changes,
            yaku_point_infos,
        ]


def determine_ura_doras_list(state: mjproto.State) -> List:
    if len(state.terminal.wins) == 0:  # あがり者の有無でウラどらが表示されるかどうかが決まる
        return []
    elif 1 not in state.terminal.wins[0].yakus:  # リーチがかかっていないと、上がって裏ドラが表示されない.
        return []
    else:
        return [_change_tile_fmt(i) for i in state.ura_doras]


# ここを実装
def mjproto_to_mjscore(state: mjproto.State) -> str:
    round: int = state.init_score.round
    honba: int = state.init_score.honba
    riichi: int = state.init_score.riichi
    doras: List[int] = [_change_tile_fmt(i) for i in state.doras]
    ura_doras = determine_ura_doras_list(state)
    init_score: List[int] = [i for i in state.init_score.ten]
    log = [[round, honba, riichi], init_score, doras, ura_doras]
    absolute_pos = [
        mjproto.ABSOLUTE_POS_INIT_EAST,
        mjproto.ABSOLUTE_POS_INIT_SOUTH,
        mjproto.ABSOLUTE_POS_INIT_WEST,
        mjproto.ABSOLUTE_POS_INIT_NORTH,
    ]
    for abs_pos in absolute_pos:
        log.append(sort_init_hand(_change_tiles_fmt(state.private_infos[abs_pos].init_hand)))
        log.append(
            parse_draws(state.private_infos[abs_pos].draws, state.event_history.events, abs_pos)
        )
        log.append(parse_discards(state.event_history.events, abs_pos))

    log.append(parse_terminal(state))
    d: Dict = {"title": [], "name": [], "rule": [], "log": [log]}
    return json.dumps(d)
