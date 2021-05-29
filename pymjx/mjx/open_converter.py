# postpone type hint evaluation or doctest fails
from __future__ import annotations

from typing import List

import mjxproto
from mjx.const import RelativePos


def open_event_type(bits: int) -> mjxproto.EventTypeValue:
    """
    >>> open_event_type(47723) == mjxproto.EVENT_TYPE_PON
    True
    >>> open_event_type(49495) == mjxproto.EVENT_TYPE_CHI
    True
    >>> open_event_type(28722) == mjxproto.EVENT_TYPE_ADDED_KAN
    True
    """
    if 1 << 2 & bits:
        return mjxproto.EVENT_TYPE_CHI
    elif 1 << 3 & bits:
        return mjxproto.EVENT_TYPE_PON
    elif 1 << 4 & bits:
        return mjxproto.EVENT_TYPE_ADDED_KAN
    else:
        if RelativePos.SELF == bits & 3:
            return mjxproto.EVENT_TYPE_CLOSED_KAN
        else:
            return mjxproto.EVENT_TYPE_OPEN_KAN


def open_from(bits: int) -> RelativePos:
    """
    >>> open_from(51306) == RelativePos.MID  # 対面
    True
    >>> open_from(49495) == RelativePos.LEFT  # 上家
    True
    >>> open_from(28722) == RelativePos.MID  # 加槓
    True
    """

    event_type = open_event_type(bits)
    if event_type == mjxproto.EVENT_TYPE_CHI:
        return RelativePos.LEFT
    elif (
        event_type == mjxproto.EVENT_TYPE_PON
        or event_type == mjxproto.EVENT_TYPE_OPEN_KAN
        or event_type == mjxproto.EVENT_TYPE_ADDED_KAN
    ):
        return RelativePos(bits & 3)
    else:
        return RelativePos.SELF


def _min_tile_chi(bits: int) -> int:
    x = (bits >> 10) // 3  # 0~21
    min_tile = (x // 7) * 9 + x % 7  # 0~33 base 9
    return min_tile


def is_stolen_red(bits: int, stolen_tile_kind) -> bool:  # TODO: test  さらに小さい関数を作るか否か考えるべし
    """
    >>> is_stolen_red(51306, 33)
    False
    >>> is_stolen_red(51247, 22)
    True
    """
    fives = [4, 13, 22]
    reds = [14, 52, 88]
    event_type = open_event_type(bits)
    if stolen_tile_kind not in fives:
        return False

    if event_type == mjxproto.EVENT_TYPE_CHI:
        stolen_tile_mod3 = (bits >> 10) % 3  # 鳴いた牌のindex
        stolen_tile_id_mod4 = (bits >> (3 + 2 * stolen_tile_mod3)) % 4  # 鳴いた牌のi
        return stolen_tile_id_mod4 == 0  # 鳴いた牌のid mod 4=0→赤
    elif event_type == mjxproto.EVENT_TYPE_PON or event_type == mjxproto.EVENT_TYPE_ADDED_KAN:
        unused_id_mod4 = (bits >> 5) % 4  # 未使用牌のid mod 4
        stolen_tile_mod3 = (bits >> 9) % 3  # 鳴いた牌のindex
        return unused_id_mod4 != 0 and stolen_tile_mod3 == 0  # 未使用牌が赤でなく、鳴いた牌のインデックスが0の時→赤
    else:
        return (bits >> 8) in reds


def is_unused_red(bits: int):
    unused_id_mod4 = (bits >> 5) % 4
    return unused_id_mod4 == 0


def has_red_chi(bits: int) -> bool:  # TODO テストgit
    min_starts_include5_mod9 = [2, 3, 4]
    min_tile = _min_tile_chi(bits)
    if min_tile % 9 not in min_starts_include5_mod9:
        return False
    else:
        start_from3 = min_tile % 9 == 2  # min_tile で場合分け
        start_from4 = min_tile % 9 == 3
        start_from5 = min_tile % 9 == 4
        if start_from3:  # 3から始まる→3番目の牌のid mod 4 =0 →赤
            return (bits >> 7) % 4 == 0
        elif start_from4:
            return (bits >> 5) % 4 == 0
        elif start_from5:
            return (bits >> 3) % 4 == 0
        else:
            assert False


# TODO テスト ポンとカカンは未使用牌が赤かどうかで鳴牌に赤があるか判断
def has_red_pon_kan_added(bits: int) -> bool:
    fives = [4, 13, 22, 51, 52, 53]
    stolen_tile_kind = open_stolen_tile_type(bits)
    if stolen_tile_kind in fives:
        unused_id_mod4 = (bits >> 5) % 4
        return unused_id_mod4 != 0
    else:
        return False


def has_red_kan_closed_kan_opend(bits: int) -> bool:
    fives = [4, 13, 22, 51, 52, 53]
    stolen_tile_kind = open_stolen_tile_type(bits)
    return stolen_tile_kind in fives


def has_red(bits: int) -> bool:
    """
    >>> has_red(52503)  # 赤５を含むチー
    True
    """
    event_type = open_event_type(bits)
    if event_type == mjxproto.EVENT_TYPE_CHI:
        return has_red_chi(bits)
    elif event_type == mjxproto.EVENT_TYPE_PON or event_type == mjxproto.EVENT_TYPE_ADDED_KAN:
        return has_red_pon_kan_added(bits)
    else:
        return has_red_kan_closed_kan_opend(bits)  # ダイミンカンとアンカンは必ず赤を含む


def transform_red_stolen(bits: int, stolen_tile: int) -> int:
    red_dict = {4: 51, 13: 52, 22: 53}  # openの5:mjscoreの赤５
    if is_stolen_red(bits, stolen_tile):
        return red_dict[stolen_tile]
    else:
        return stolen_tile


def transform_red_open(bits: int, open: List[int], event_type) -> List[int]:
    """
    >>> transform_red_open(52503, [21, 22, 23], mjxproto.EVENT_TYPE_CHI)
    [21, 53, 23]
    """
    red_dict = {4: 51, 13: 52, 22: 53}
    fives = [4, 13, 22]
    if not has_red(bits):
        return open
    if event_type == mjxproto.EVENT_TYPE_CHI:
        return [red_dict[i] if i in fives else i for i in open]
    else:
        open[-1] = red_dict[open[-1]]
        return open


def open_stolen_tile_type(bits: int) -> int:
    """
    >>> open_stolen_tile_type(51306)
    33
    >>> open_stolen_tile_type(49495)
    20
    >>> open_stolen_tile_type(28722)
    18
    >>> open_stolen_tile_type(31744)  #  暗槓　白
    31
    """
    event_type = open_event_type(bits)
    if event_type == mjxproto.EVENT_TYPE_CHI:
        min_tile = _min_tile_chi(bits)
        stolen_tile_kind = min_tile + (bits >> 10) % 3
        return transform_red_stolen(bits, stolen_tile_kind)
    elif event_type == mjxproto.EVENT_TYPE_PON or event_type == mjxproto.EVENT_TYPE_ADDED_KAN:
        stolen_tile_kind = (bits >> 9) // 3
        return transform_red_stolen(bits, stolen_tile_kind)
    else:
        stolen_tile_kind = (bits >> 8) // 4
        return transform_red_stolen(bits, stolen_tile_kind)


def open_tile_types(bits: int) -> List[int]:
    """
    >>> open_tile_types(51306)  # Pon rd
    [33, 33, 33]
    >>> open_tile_types(49495)  # Chi s3s4s5
    [20, 21, 22]
    >>> open_tile_types(28722)  # 加槓 s1
    [18, 18, 18, 18]
    >>> open_tile_types(31744)  # 暗槓　白
    [31, 31, 31, 31]
    """
    reds = [51, 52, 53]
    red_five_dict = {51: 4, 52: 13, 53: 22}
    event_type = open_event_type(bits)
    if event_type == mjxproto.EVENT_TYPE_CHI:
        min_tile = _min_tile_chi(bits)
        open = [min_tile, min_tile + 1, min_tile + 2]
        return transform_red_open(bits, open, event_type)
    stolen_tile_kind = open_stolen_tile_type(bits)
    if stolen_tile_kind in reds:  # 赤だった場合はopenのformatに置換しないと、複数の同じ赤を持ったopenが生成される。
        stolen_tile_kind = red_five_dict[stolen_tile_kind]
    else:
        None
    if event_type == mjxproto.EVENT_TYPE_PON:
        open = [stolen_tile_kind] * 3
        return transform_red_open(bits, open, event_type)
    else:
        open = [stolen_tile_kind] * 4
        return transform_red_open(bits, open, event_type)


def change_open_tile_fmt(
    tile_in_open_fmt: int,
) -> int:  # tile_in_open 0~33 tile_in_score 11~19, 21~29, 31~39,41~47
    reds_in_score = [51, 52, 53]
    if tile_in_open_fmt in reds_in_score:
        return tile_in_open_fmt
    else:
        tile_in_score = 10 + 10 * (tile_in_open_fmt // 9) + (tile_in_open_fmt % 9 + 1)
        return tile_in_score


def change_open_tiles_fmt(tile_ids_in_open: List[int]) -> List[int]:
    """
    >>> change_open_tiles_fmt([21, 22, 23])
    [34, 35, 36]
    """
    scores = list(map(change_open_tile_fmt, tile_ids_in_open))
    return scores
