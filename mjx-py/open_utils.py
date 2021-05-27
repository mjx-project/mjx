from __future__ import annotations
from typing import List
import mjxproto
from converter import TileUnitType, FromWho

MASK_CHI_OFFSET = [0b0000000000011000, 0b0000000001100000, 0b0000000110000000]
MASK_PON_UNUSED_OFFSET = 0b0000000001100000


def open_event_type(bits: int) -> TileUnitType:
    """
    >>> open_event_type(47723) == TileUnitType.PON
    True
    >>> open_event_type(49495) == TileUnitType.CHI
    True
    >>> open_event_type(28722) == TileUnitType.ADDED_KAN
    True
    """
    if 1 << 2 & bits:
        return TileUnitType.CHI
    elif 1 << 3 & bits:
        return TileUnitType.PON
    elif 1 << 4 & bits:
        return TileUnitType.ADDED_KAN
    else:
        if 0 == bits & 3:
            return TileUnitType.CLOSED_KAN
        else:
            return TileUnitType.OPEN_KAN


def open_from(bits: int) -> FromWho:
    """
    >>> open_from(51306) == RelativePos.MID  # 対面
    True
    >>> open_from(49495) == RelativePos.LEFT  # 上家
    True
    >>> open_from(28722) == RelativePos.MID  # 加槓
    True
    """

    event_type = open_event_type(bits)
    if event_type == TileUnitType.CHI:
        return FromWho.LEFT
    elif (
        event_type == TileUnitType.PON
        or event_type == TileUnitType.KAN
        or event_type == TileUnitType.KAN
    ):
        return bits & 3
    else:
        return FromWho.SELF


def chi_ids_(bits: int) -> List[int]:
    """
    チーに使われた牌のidを返す関数
    """
    min_type_base = (bits >> 10) // 3
    min_type = (min_type_base // 7) * 9 + min_type_base % 7
    return [
        (min_type + i) * 4 + ((bits & MASK_CHI_OFFSET[i]) >> (2 * i + 3))
        for i in range(3)
    ]


def pon_ids_(bits: int) -> List[int]:
    """
    ポンに使われた牌のidを返す関数
    """
    ids = []
    for i in range(3):
        type = (bits >> 9) // 3
        unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5
        if i >= unused_offset:
            i = i + 1
        ids.append(type * 4 + i)
    return ids


def oepn_kan_ids_(bits: int) -> List[int]:
    """
    大明槓に使われた牌のidを返す関数
    """
    return [((bits >> 8) // 4) * 4 + i for i in range(4)]


def closed_kan_ids_(bits: int) -> List[int]:
    """
    暗槓に使われた牌のidを返す関数
    """
    return [((bits >> 8) // 4) * 4 + i for i in range(4)]


def added_kan_ids_(bits: int) -> List[int]:
    """
    加槓に使われた牌のidを返す関数
    """
    return [((bits >> 9) // 3) * 4 + i for i in range(4)]


def open_tile_ids(bits: int) -> List[int]:
    """
    各鳴きで使われた牌のidを返す関数
    以下、各鳴きに対するdoctest
    >>> open_tile_ids(49495)  # chi
    [82, 86, 90]
    >>> open_tile_ids(47723)  # pon
    [124, 125, 126]
    >>> open_tile_ids(1793)  # open_kan
    [4, 5, 6, 7]
    >>> open_tile_ids(2048)  # closed_kan
    [8, 9, 10, 11]
    >>> open_tile_ids(530)  # added_kan
    [0, 1, 2, 3]
    """
    event_type = open_event_type(bits)
    if event_type == TileUnitType.CHI:
        return chi_ids_(bits)
    elif event_type == TileUnitType.PON:
        return pon_ids_(bits)
    elif event_type == TileUnitType.OPEN_KAN:
        return oepn_kan_ids_(bits)
    elif event_type == TileUnitType.CLOSED_KAN:
        return closed_kan_ids_(bits)
    else:
        return added_kan_ids_(bits)
