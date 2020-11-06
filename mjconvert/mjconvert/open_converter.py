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


def open_tile_types(bits: int) -> List[int]:
    
    >>> open_tile_types(51306)  # Pon rd
    [33, 33, 33]
    >>> open_tile_types(49495)  # Chi s3s4s5
    [20, 21, 22]
    >>> open_tile_types(28722)  # 加槓 s
    >> [18, 18, 18, 18]
    

    return []



def open_from(bits: int) -> mj_pb2.RelativePos:
    """
    >>> open_from(51306) == mj_pb2.RELATIVE_POS_MID  # 対面
    True
    >>> open_from(49495) == mj_pb2.RELATIVE_POS_LEFT  # 上家
    True
    >>> open_from(28722) == mj_pb2.RELATIVE_POS_SELF  # 加槓
    True
    """
    event_type = open_event_type(bits)
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        return mj_pb2.RELATIVE_POS_LEFT
    elif event_type == mj_pb2.EVENT_TYPE_PON or event_type == mj_pb2.EVENT_TYPE_KAN_OPENED:
        if 3 & bits:
            return mj_pb2.RELATIVE_POS_LEFT
        elif 2 & bits:
            return mj_pb2.RELATIVE_POS_MID
        elif 1 & bits:
            return mj_pb2.RELATIVE_POS_RIGHT
    elif event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
        return mj_pb2.RELATIVE_POS_SELF
    else:
        return mj_pb2.RELATIVE_POS_SELF


ids = [[i, i + 1, i + 2, i + 3] for i in range(0, 136, 4)]  # こいつの位置をどうするか


# チーで晒された牌のidのmod4をを小さい順に並べたものを返す
def mod4_for_chi(bits: int):
    return [(bits >> 3) & 3, (bits >> 5) & 3, (bits >> 7) & 3]


# 　ポンカスのidのmod4
def mod4_for_pon(bits: int):
    return (bits >> 5) & 3


# 　34までの牌の種類を表す数字を受け取ってmjproto形式で対応するidとしてあり得るものを出力
def possible_ids(tile: int) -> List[int]:
    return ids[tile]

def open_tile_info_for_chi(bits: int):
    x = (bits >> 10) // 3
    min_tile = x if x <= 6 else (x + 2 if x <= 13 else x + 4)  # 8m,9m,8p,9pが抜けているので34種類でのインデックスに対応させる
    stolen_tile_kind = min_tile + (bits >> 10) % 3
    stolen_tile_mod3 = (bits >> 10) % 3  # 3種類のうち鳴いた牌のindex
    pos_ids = possible_ids(stolen_tile_kind)
    stolen_id_mod4 = mod4_for_chi(bits)[stolen_tile_mod3]
    stolen_id = pos_ids.pop(stolen_id_mod4)
    open_ids = pos_ids
    return [stolen_id, open_ids]


def open_tile_info_for_pon(bits: int):
    stolen_tile_kind = (bits >> 9) // 3
    stolen_tile_mod3 = (bits >> 9) % 3  # 3つのidのうち鳴いた牌のindex
    unused_id_mod4 = mod4_for_pon(bits)
    pos_ids = possible_ids(stolen_tile_kind)
    del pos_ids[unused_id_mod4]
    stolen_id = pos_ids.pop(stolen_tile_mod3)
    open_ids = pos_ids
    return [stolen_id, open_ids]

#def open_tile_info_for_kan_added(bits: int):



def open_stolen_tile_type(bits: int) -> int:
    """
    >>> open_stolen_tile_type(51306)
    33
    >>> open_stolen_tile_type(49495)
    20
    >>> open_stolen_tile_type(28722)
    18
    """
    # chiとそれ以外のアクションでは取得できる牌の種類が異なる。
    event_type = open_event_type(bits)
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        stolen_id = open_tile_info_for_chi(bits)[0]
        return stolen_id
    elif event_type == mj_pb2.EVENT_TYPE_PON:
        stolen_id = open_tile_info_for_pon(bits)[0]
        return stolen_id
    #elif event_type == mj_pb2.EVENT_TYPE_KAN_CLOSED or event_type == mj_pb2.EVENT_TYPE_KAN_OPENED:
        #stolen_tile_kind =


def open_tile_types(bits: int) -> List[int]:
    """
    >> > open_tile_types(51306)  # Pon rd
    [33, 33, 33]
    >> > open_tile_types(49495)  # Chi s3s4s5
    [20, 21, 22]
    >> > open_tile_types(28722)  # 加槓 s
    >> [18, 18, 18, 18]
    """


    return []


print(open_stolen_tile_type(49495))
