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


def _min_tile_chi(bits: int) -> int:
    x = (bits >> 10) // 3  # 0~21
    min_tile = (x // 7) * 9 + x % 7  # 0~33 base 9
    return min_tile


def transform_red_stolen_tile(bits: int, stolen_tile_kind: int) -> int:  # to_do: test  さらに小さい関数を作るか否か考えるべし
    fives = [4, 14, 22]
    reds = [14, 52, 88]
    reds_dict = {4: 51, 14: 52, 22: 53}
    event_type = open_event_type(bits)
    if stolen_tile_kind in fives:
        if event_type == mj_pb2.EVENT_TYPE_CHI:
            stolen_tile_mod3 = (bits >> 10) % 3  # 鳴いた牌のindex
            stolen_tile_id_mod4 = bits >> (3 + 2 * stolen_tile_mod3) & 3  # 鳴いた牌のid mod 4
            if stolen_tile_id_mod4 == 0:  # 鳴いた牌のid mod 4=0→赤
                return reds_dict[stolen_tile_kind]
            else:
                return stolen_tile_kind
        elif event_type == mj_pb2.EVENT_TYPE_PON or event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
            unused_id_mod4 = (bits >> 5) & 3  # 未使用牌のid mod 4
            stolen_tile_mod3 = (bits >> 9) % 3  # 鳴いた牌のindex
            if unused_id_mod4 != 0 and stolen_tile_mod3 == 0:  # 未使用牌が赤でなく、鳴いた牌のインデックスが0の時→赤
                return reds_dict[stolen_tile_kind]
            else:
                return stolen_tile_kind
        else:
            if (bits >> 8) in reds:
                return reds_dict[stolen_tile_kind]
            else:
                return stolen_tile_kind
    else:
        return stolen_tile_kind


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
    fives = [4, 14, 22]
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        min_tile = _min_tile_chi(bits)
        stolen_tile_kind = min_tile + (bits >> 10) % 3
        return transform_red_stolen_tile(bits, stolen_tile_kind)
    elif event_type == mj_pb2.EVENT_TYPE_PON or event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
        stolen_tile_kind = (bits >> 9) // 3
        return transform_red_stolen_tile(bits, stolen_tile_kind)
    else:
        stolen_tile_kind = (bits >> 8) // 4  # to_do:テスト
        return transform_red_stolen_tile(bits, stolen_tile_kind)


def transform_red_open_tile(bits: int, stolen_tile_kind=0, min_tile=0) -> List[int]:  # to_do:テスト
    fives = [4, 14, 22, 51, 52, 53]
    min_starts_include5_mod9 = [2, 3, 4]
    reds_dict = {4: 51, 14: 52, 22: 53}
    event_type = open_event_type(bits)
    if stolen_tile_kind in fives or min_tile % 9 in min_starts_include5_mod9:  # 5晒した牌に含まれるかどうかで場合分け。
        if event_type == mj_pb2.EVENT_TYPE_CHI:
            start_from3 = min_tile % 9 == 2  # min_tile で場合分け
            start_from4 = min_tile % 9 == 3
            start_from5 = min_tile % 9 == 4
            if start_from3 and ((bits >> 7) & 3 == 0):  # 3から始まる→3番目の牌のid mod 4 =0 →赤
                return [min_tile, min_tile + 1, reds_dict[min_tile + 2]]
            elif start_from4 and ((bits >> 5) & 3) == 0:
                return [min_tile, reds_dict[min_tile + 1], min_tile + 2]
            elif start_from5 and ((bits >> 3) & 3 == 0):
                return [reds_dict[min_tile], min_tile + 1, min_tile + 2]
            else:
                return [min_tile, min_tile + 1, min_tile + 2]
        elif event_type == mj_pb2.EVENT_TYPE_PON:
            unused_id_mod3 = (bits >> 5) & 3
            stolen_tile_kind = (bits >> 9) // 3  # open_stolen_tile_typeで赤が吐き出された場合の処理がややこしいので再定義
            if unused_id_mod3 != 0:
                return [stolen_tile_kind, stolen_tile_kind,
                        reds_dict[stolen_tile_kind]]  # mjscoreでは同じ種類の牌のうち赤が最後に配置される。
            else:
                return [stolen_tile_kind] * 3
        elif event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
            stolen_tile_kind = (bits >> 9) // 3  # open_stolen_tile_typeで赤が吐き出された場合の処理がややこしいので再定義
            return [stolen_tile_kind, stolen_tile_kind, stolen_tile_kind, reds_dict[stolen_tile_kind]]
        else:
            stolen_tile_kind = (bits >> 8) // 4
            return [stolen_tile_kind, stolen_tile_kind, stolen_tile_kind, reds_dict[stolen_tile_kind]]
    else:
        if event_type == mj_pb2.EVENT_TYPE_CHI:
            return [min_tile, min_tile + 1, min_tile + 2]
        elif event_type == mj_pb2.EVENT_TYPE_PON:
            return [stolen_tile_kind] * 3
        else:
            return [stolen_tile_kind] * 4


def open_tile_types(bits: int) -> List[int]:
    """
    >>> open_tile_types(51306)  # Pon rd
    [33, 33, 33]
    >>> open_tile_types(49495)  # Chi s3s4s5
    [20, 21, 22]
    >>> open_tile_types(28722)  # 加槓 s1
    [18, 18, 18, 18]
    """
    fives = [4, 14, 22, 51, 52, 53]  # open_stolen_tile_type()から赤が帰ってきた場合に備えて
    event_type = open_event_type(bits)
    if event_type == mj_pb2.EVENT_TYPE_CHI:
        min_tile = _min_tile_chi(bits)
        return transform_red_open_tile(bits, min_tile=min_tile)
    elif event_type == mj_pb2.EVENT_TYPE_PON:
        stolen_tile_kind = open_stolen_tile_type(bits)
        return transform_red_open_tile(bits, stolen_tile_kind=stolen_tile_kind)
    elif event_type == mj_pb2.EVENT_TYPE_KAN_ADDED:
        stolen_tile_kind = open_stolen_tile_type(bits)
        return transform_red_open_tile(bits, stolen_tile_kind=stolen_tile_kind)
    else:
        stolen_tile_kind = (bits >> 8) // 4  # to_do:テスト
        return transform_red_open_tile(bits, stolen_tile_kind=stolen_tile_kind)

print(open_tile_types(51206))