from typing import List, Optional

import mjxproto
from mjx import EventType, Observation, Tile


def produce(obs: mjxproto.Observation) -> List[List[int]]:
    feature = []
    for func in [
        current_hand,
        target_tile,
        under_riichis,
        discarded_tiles,
        discarded_from_hand,
        opened_tiles,
        shanten,
        dealer,
        doras,
        effective_draws,
        effective_discards,
        ignored_tiles,
        kyotaku,
        rankings,
        round,
        honba,
        dora_num_in_hand,
        dora_num_of_target,
    ]:
        feature.extend(func(obs))
    return feature


def current_hand(obs: mjxproto.Observation):
    # feat[i][t] := obs の観測者がtile t を(i+1)枚以上持っているか
    #               (i=0,1,2,3)
    # feat[i+4] := obs の観測者がsuit i の赤ドラを持っているか
    #              (i=0,1,2)
    feat = [[0] * 34 for _ in range(4 + 3)]
    hand = obs.curr_hand()
    for tile_type, cnt in enumerate(hand.closed_tile_types()):
        for i in range(cnt):
            feat[i][tile_type] = 1
    for tile in hand.closed_tiles():
        if tile.is_red():
            feat[4 + (tile.type() // 9)] = [1] * 34
    return feat

    @staticmethod
    def target_tile(obs: Observation):
        # feat[0] := target tile (one-hot)
        # feat[1] := target tile が赤ドラかどうか
        feat = [[0] * 34 for _ in range(2)]

        def _target_tile() -> Optional[Tile]:
            events = obs.events()
            if len(events) == 0:
                return None

            last_event = events[-1]
            if last_event.type() in [EventType.DISCARD, EventType.TSUMOGIRI]:
                return last_event.tile()

            elif last_event.type() == EventType.DRAW:
                return obs.draw_history()[-1]
            elif last_event.type() == EventType.ADDED_KAN:
                return last_event.open().last_tile()
            else:
                return None

        target = _target_tile()
        if target is None:
            return feat

        feat[0][target.type()] = 1
        if target.is_red():
            feat[1] = [1] * 34

        return feat

    @staticmethod
    def under_riichis(obs: Observation):
        # 自家: feat[0]
        # 下家: feat[1]
        # 対面: feat[2]
        # 上家: feat[3]
        feat = [[0] * 34 for _ in range(4)]

        for i in range(4):
            if any([e.type() == EventType.RIICHI and e.who() == i for e in obs.events()]):
                feat[(i - obs.who() + 4) % 4] = [1] * 34

        return feat

    @staticmethod
    def discarded_tiles(obs: Observation):
        discarded = [[] for _ in range(4)]

        for e in obs.events():
            if e.type() != EventType.DISCARD and e.type() != EventType.TSUMOGIRI:
                continue
            discarded[e.who()].append(e.tile().type())

        # 自家: feat[0:3]
        # 下家: feat[3:6]
        # 対面: feat[6:9]
        # 上家: feat[9:12]
        feat = [[0] * 34 for _ in range(3 * 4)]

        for p in range(4):
            q = (p - obs.who() + 4) % 4
            for t in discarded[p][:6]:
                feat[q * 3][t] = 1
            for t in discarded[p][6:12]:
                feat[q * 3 + 1][t] = 1
            for t in discarded[p][12:]:
                feat[q * 3 + 2][t] = 1

        return feat

    @staticmethod
    def discarded_from_hand(obs: Observation):
        discarded = [[] for _ in range(4)]

        for e in obs.events():
            if e.type() != EventType.DISCARD:
                continue
            discarded[e.who()].append(e.tile().type())

        # 自家: feat[0:3]
        # 下家: feat[3:6]
        # 対面: feat[6:9]
        # 上家: feat[9:12]
        feat = [[0] * 34 for _ in range(3 * 4)]

        for p in range(4):
            q = (p - obs.who() + 4) % 4
            for t in discarded[p][:6]:
                feat[q * 3][t] = 1
            for t in discarded[p][6:12]:
                feat[q * 3 + 1][t] = 1
            for t in discarded[p][12:]:
                feat[q * 3 + 2][t] = 1

        return feat

    @staticmethod
    def opened_tiles(obs: Observation):
        # feat[q*7 + i][t] := player q の鳴きにtile t が(i+1)枚以上含まれているか
        #                     (i=0,1,2,3)
        # feat[q*7 + i+4] := player q の鳴きにsuit i の赤ドラが含まれているか
        #                     (i=0,1,2)
        # 自家: feat[0:7]
        # 下家: feat[7:14]
        # 対面: feat[14:21]
        # 上家: feat[21:28]

        opened = [[0] * 34 for i in range(4)]
        is_red = [[0] * 3 for i in range(4)]

        for e in obs.events():
            if e.type() in [
                EventType.CHI,
                EventType.PON,
                EventType.CLOSED_KAN,
                EventType.OPEN_KAN,
            ]:
                for t in e.open().tiles():
                    opened[e.who()][t.type()] += 1
                    if t.is_red():
                        is_red[e.who()][t.type() // 9] = 1
            # KAN_ADDED は last_tile だけ追加.
            if e.type() == EventType.ADDED_KAN:
                t = e.open().last_tile()
                opened[e.who()][t.type()] += 1
                if t.is_red():
                    is_red[e.who()][t.type() // 9] = 1

        feat = [[0] * 34 for _ in range(7 * 4)]

        for p in range(4):
            q = (p - obs.who() + 4) % 4
            for tile_type in range(34):
                for i in range(opened[p][tile_type]):
                    feat[q * 7 + i][tile_type] = 1
                for i in range(3):
                    if is_red[p][i]:
                        feat[q * 7 + i + 4] = [1] * 34

        return feat

    @staticmethod
    def dealer(obs: Observation):
        feat = [[0] * 34 for _ in range(4)]
        feat[(obs.dealer() - obs.who() + 4) % 4] = [1] * 34
        return feat

    @staticmethod
    def doras(obs: Observation):
        dora_cnt = [0] * 34
        for dora in obs.doras():
            dora_cnt[dora] += 1

        feat = [[0] * 34 for _ in range(4)]
        for dora in range(34):
            for i in range(dora_cnt[dora]):
                feat[i][dora] = 1
        return feat

    @staticmethod
    def shanten(obs: Observation):
        # feat[i] := shanten数が i以上か
        feat = [[0] * 34 for _ in range(7)]
        for i in range(obs.curr_hand().shanten_number() + 1):
            feat[i] = [1] * 34
        return feat

    @staticmethod
    def effective_discards(obs: Observation):
        hand = obs.curr_hand()
        feat = [0] * 34
        tile_types = hand.effective_discard_types()
        for tt in tile_types:
            feat[tt] = 1
        return [feat]

    @staticmethod
    def effective_draws(obs: Observation):
        hand = obs.curr_hand()
        feat = [0] * 34
        tile_types = hand.effective_draw_types()
        for tt in tile_types:
            feat[tt] = 1
        return [feat]

    @staticmethod
    def ignored_tiles(obs: Observation):
        ignored = [set() for i in range(4)]

        for e in obs.events():
            if e.type() != EventType.DISCARD and e.type() != EventType.TSUMOGIRI:
                continue
            for i in range(4):
                if i != e.who():
                    continue
                ignored[i].add(e.tile().type())

            if e.type() == EventType.DISCARD:
                ignored[i] = set()

        feat = [[0] * 34 for _ in range(4)]

        for p in range(4):
            q = (p - obs.who() + 4) % 4
            for t in ignored[p]:
                feat[q][t] = 1

        return feat

    @staticmethod
    def kyotaku(obs: Observation):
        # feat[i] := 供託がi+1 本以上あるか
        feat = [[0] * 34 for _ in range(5)]
        for i in range(min(obs.kyotaku(), 5)):
            feat[i] = [1] * 34
        return feat

    @staticmethod
    def rankings(obs: Observation):
        # feat[q*3 + i] := player q の現在の着順(0,1,2,3) がi+1 以上か
        feat = [[0] * 34 for _ in range(3 * 4)]
        for p, r in enumerate(obs.rankings()):
            q = (p + 4 - obs.who()) % 4
            for i in range(r):
                feat[q * 3 + i] = [1] * 34
        return feat

    @staticmethod
    def round(obs: Observation):
        # 東一局:0, ..., 南四局:7
        # feat[i] := 現在のroundが i+1以上か
        feat = [[0] * 34 for _ in range(7)]
        for i in range(min(obs.round(), 7)):
            feat[i] = [1] * 34
        return feat

    @staticmethod
    def honba(obs: Observation):
        # feat[i] := 現在の本場がi+1 以上か
        feat = [[0] * 34 for _ in range(5)]
        for i in range(min(obs.honba(), 5)):
            feat[i] = [1] * 34
        return feat

    @staticmethod
    def dora_num_of_target(obs: Observation):
        def _target_tile() -> Optional[Tile]:
            events = obs.events()
            if len(events) == 0:
                return None

            last_event = events[-1]
            if last_event.type() in [EventType.DISCARD, EventType.TSUMOGIRI]:
                return last_event.tile()

            elif last_event.type() == EventType.DRAW:
                return obs.draw_history()[-1]
            elif last_event.type() == EventType.ADDED_KAN:
                return last_event.open().last_tile()
            else:
                return None

        feat = [[0] * 34 for _ in range(4)]
        # feat[i] := dora num of target tile >= i+1 ?

        tt = _target_tile()
        if tt is None:
            return feat

        dora_num = 0
        if tt.is_red():
            dora_num += 1

        for dora in obs.doras():
            if tt.type() == dora:
                dora_num += 1

        for i in range(dora_num):
            feat[i] = [1] * 34

        return feat

    @staticmethod
    def dora_num_in_hand(obs: Observation):

        dora_num = 0
        for tile in obs.curr_hand().closed_tiles():
            if tile.is_red():
                dora_num += 1
            for dora in obs.doras():
                if tile.type() == dora:
                    dora_num += 1

        for meld in obs.curr_hand().opens():
            for tile in meld.tiles():
                if tile.is_red():
                    dora_num += 1
                for dora in obs.doras():
                    if tile.type() == dora:
                        dora_num += 1

        if dora_num > 13:
            dora_num = 13

        feat = [[0] * 34 for _ in range(13)]
        # feat[i] := dora num of target tile >= i+1 ?

        for i in range(dora_num):
            feat[i] = [1] * 34

        return feat
