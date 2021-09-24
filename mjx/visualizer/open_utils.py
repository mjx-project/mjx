from __future__ import annotations

from typing import List

from mjx.visualizer.converter import FromWho, TileUnitType

MASK_CHI_OFFSET = [0b0000000000011000, 0b0000000001100000, 0b0000000110000000]
MASK_PON_UNUSED_OFFSET = 0b0000000001100000


def open_event_type(bits: int) -> TileUnitType:
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

    event_type = open_event_type(bits)
    if event_type == TileUnitType.CHI:
        return FromWho.LEFT
    elif (
        event_type == TileUnitType.PON
        or event_type == TileUnitType.OPEN_KAN
        or event_type == TileUnitType.ADDED_KAN
    ):
        if bits & 3 == 1:
            return FromWho.RIGHT
        if bits & 3 == 2:
            return FromWho.MID
        if bits & 3 == 3:
            return FromWho.LEFT
    else:
        return FromWho.SELF


def chi_ids_(bits: int) -> List[int]:
    """
    チーに使われた牌のidを返す関数
    """
    min_type_base = (bits >> 10) // 3
    min_type = (min_type_base // 7) * 9 + min_type_base % 7
    return [(min_type + i) * 4 + ((bits & MASK_CHI_OFFSET[i]) >> (2 * i + 3)) for i in range(3)]


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
