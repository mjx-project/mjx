from typing import List
from mjconvert import mj_pb2


def open_event_type(bits: int) -> mj_pb2.EventType:
    """
    >>> open_event_type(47723) == mj_pb2.EVENT_TYPE_PON
    True
    >>> open_event_type(49495) == mj_pb2.EVENT_TYPE_CHI
    True
    >>> open_event_type(28722) == mj_pb2.EVENT_TYPE_KAN_ADDED
    True
    """
    if 1 << 2 & bits:
        return mj_pb2.EVENT_TYPE_CHI
    elif 1 << 3 & bits:
        return mj_pb2.EVENT_TYPE_PON
    elif 1 << 4 & bits:
        return mj_pb2.EVENT_TYPE_KAN_ADDED
    else:
        if mj_pb2.RELATIVE_POS_SELF == bits & 3:
            return mj_pb2.EVENT_TYPE_KAN_CLOSED
        else:
            return mj_pb2.EVENT_TYPE_KAN_OPENED


def open_from(bits: int) -> mj_pb2.RelativePos:
    """
    >>> open_from(51306) == mj_pb2.RELATIVE_POS_MID  # 対面
    True
    >>> open_from(49495) == mj_pb2.RELATIVE_POS_LEFT  # 上家
    True
    >>> open_from(28722) == mj_pb2.RELATIVE_POS_MID  # 加槓
    True
    """
    event_type = open_event_type(bits)
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        return mj_pb2.RELATIVE_POS_LEFT
    elif event_type == mj_pb2.EVENT_TYPE_PON or event_type == mj_pb2.EVENT_TYPE_KAN_OPENED or event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
        if 3 == (3 & bits):
            return mj_pb2.RELATIVE_POS_LEFT
        elif 2 == (2 & bits) & bits:
            return mj_pb2.RELATIVE_POS_MID
        elif 1 == (1 & bits) & bits:
            return mj_pb2.RELATIVE_POS_RIGHT
    else:
        return mj_pb2.RELATIVE_POS_SELF


def open_stolen_tile_type(bits: int) -> int:
    """
    >>> open_stolen_tile_type(51306)
    33
    >>> open_stolen_tile_type(49495)
    20
    >>> open_stolen_tile_type(28722)
    18
    """
    event_type = open_event_type(bits)
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        x = (bits >> 10) // 3
        min_tile = x if x <= 6 else (x + 2 if x <= 13 else x + 4)  # 8m,9m,8p,9pが抜けているので34種類でのインデックスに対応させる
        stolen_tile_kind = min_tile + (bits >> 10) % 3
        return stolen_tile_kind
    elif event_type == mj_pb2.EVENT_TYPE_PON or event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
        stolen_tile_kind = (bits >> 9) // 3
        return stolen_tile_kind


def open_tile_types(bits: int) -> List[int]:
    """
    >>> open_tile_types(51306)  # Pon rd
    [33, 33, 33]
    >>> open_tile_types(49495)  # Chi s3s4s5
    [20, 21, 22]
    >>> open_tile_types(28722)  # 加槓 s1
    >> [18, 18, 18, 18]
    """
    event_type = open_event_type(bits)
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        x = (bits >> 10) // 3
        min_tile = x if x <= 6 else (x + 2 if x <= 13 else x + 4)  # 8m,9m,8p,9pが抜けているので34種類でのインデックスに対応させる
        return [min_tile, min_tile + 1, min_tile + 2]
    elif event_type == mj_pb2.EVENT_TYPE_PON:
        stolen_tile_kind = (bits >> 9) // 3
        return [stolen_tile_kind] * 3
    elif event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
        stolen_tile_kind = (bits >> 9) // 3
        return [stolen_tile_kind] * 4



