import mjxproto
from mjconvert.open_converter import open_event_type
from typing import Dict, List

MASK_CHI_OFFSET = [0b0000000000011000, 0b0000000001100000, 0b0000000110000000]
MASK_PON_UNUSED_OFFSET = 0b0000000001100000


def open_additional_tile_ids(bits: int) -> List[int]:
    """
    chi,
    pon,
    open_kan,
    closed_kan,
    added_kan
    >>> open_additional_tile_ids(49495)
    [82, 86, 90]
    >>> open_additional_tile_ids(47723)
    [124, 125, 126]
    >>> open_additional_tile_ids(1793)
    [4, 5, 6, 7]
    >>> open_additional_tile_ids(2048)
    [8, 9, 10, 11]
    >>> open_additional_tile_ids(530)
    [2]
    """
    event_type = open_event_type(bits)
    if event_type == mjxproto.EVENT_TYPE_CHI:
        return chi_ids(bits)
    elif event_type == mjxproto.EVENT_TYPE_PON:
        return pon_ids(bits)
    elif event_type == mjxproto.EVENT_TYPE_OPEN_KAN:
        return oepn_kan_ids(bits)
    elif event_type == mjxproto.EVENT_TYPE_CLOSED_KAN:
        return closed_kan_ids(bits)
    elif event_type == mjxproto.EVENT_TYPE_ADDED_KAN:
        type = (bits >> 9) // 3
        stolen_ix = (bits >> 9) % 3
        unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5
        if stolen_ix >= unused_offset:
            stolen_ix = stolen_ix + 1
        return [type * 4 + stolen_ix]


def open_removable_tile_ids(bits: int) -> List[int]:
    """
    chi,
    pon,
    open_kan,
    closed_kan,
    added_kan
    >>> open_removable_tile_ids(49495)
    [86, 90]
    >>> open_removable_tile_ids(47723)
    [125, 126]
    >>> open_removable_tile_ids(1793)
    [4, 5, 6]
    >>> open_removable_tile_ids(2048)
    [8, 9, 10, 11]
    >>> open_removable_tile_ids(530)
    [2]
    """
    event_type = open_event_type(bits)
    if event_type == mjxproto.EVENT_TYPE_CHI:
        tiles = chi_ids(bits)
        tiles.pop((bits >> 10) % 3)
        return tiles
    elif event_type == mjxproto.EVENT_TYPE_PON:
        tiles = pon_ids(bits)
        tiles.pop((bits >> 9) % 3)
        return tiles
    elif event_type == mjxproto.EVENT_TYPE_OPEN_KAN:
        tiles = oepn_kan_ids(bits)
        tiles.pop((bits >> 8) % 4)
        return tiles
    elif event_type == mjxproto.EVENT_TYPE_CLOSED_KAN:
        return oepn_kan_ids(bits)
    elif event_type == mjxproto.EVENT_TYPE_ADDED_KAN:
        type = (bits >> 9) // 3
        stolen_ix = (bits >> 9) % 3
        unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5
        if stolen_ix >= unused_offset:
            stolen_ix = stolen_ix + 1
        return [type * 4 + stolen_ix]


def chi_ids(bits: int) -> List[int]:
    min_type_base = (bits >> 10) // 3
    min_type = (min_type_base // 7) * 9 + min_type_base % 7
    return [(min_type + i) * 4 + ((bits & MASK_CHI_OFFSET[i]) >> (2 * i + 3)) for i in range(3)]


def pon_ids(bits: int) -> List[int]:
    ids = []
    for i in range(3):
        type = (bits >> 9) // 3
        unused_offset = (bits & MASK_PON_UNUSED_OFFSET) >> 5
        if i >= unused_offset:
            i = i + 1
        ids.append(type * 4 + i)
    return ids


def oepn_kan_ids(bits: int) -> List[int]:
    return [((bits >> 8) // 4) * 4 + i for i in range(4)]


def closed_kan_ids(bits: int) -> List[int]:
    return [((bits >> 8) // 4) * 4 + i for i in range(4)]


def added_kan_ids(bits: int) -> List[int]:
    return [((bits >> 9) // 3) * 4 + i for i in range(4)]
