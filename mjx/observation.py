from __future__ import annotations

from typing import List, Optional

import _mjx  # type: ignore
import numpy as np  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx import utils
from mjx.action import Action
from mjx.const import EventType, PlayerIdx, TileType
from mjx.event import Event
from mjx.hand import Hand
from mjx.tile import Tile
from mjx.visualizer.svg import save_svg, show_svg, to_svg


class Observation:
    def __init__(self, obs_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.Observation] = None  # type: ignore
        if obs_json is None:
            return

        self._cpp_obj = _mjx.Observation(obs_json)  # type: ignore

    def _repr_html_(self) -> None:
        observation = self.to_proto()
        return to_svg(observation, target_idx=None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            raise NotImplementedError
        raise NotImplementedError  # TODO: implement

    def who(self) -> PlayerIdx:
        assert self._cpp_obj is not None
        return self._cpp_obj.who()  # type: ignore

    def dealer(self) -> PlayerIdx:
        assert self._cpp_obj is not None
        return self._cpp_obj.dealer()  # type: ignore

    def doras(self) -> List[TileType]:
        assert self._cpp_obj is not None
        return self._cpp_obj.doras()  # type: ignore

    def curr_hand(self) -> Hand:
        assert self._cpp_obj is not None
        return Hand._from_cpp_obj(self._cpp_obj.curr_hand())  # type: ignore

    def legal_actions(self) -> List[Action]:
        assert self._cpp_obj is not None
        return [Action._from_cpp_obj(cpp_obj) for cpp_obj in self._cpp_obj.legal_actions()]  # type: ignore

    def events(self) -> List[Event]:
        assert self._cpp_obj is not None
        return [Event._from_cpp_obj(cpp_obj) for cpp_obj in self._cpp_obj.events()]  # type: ignore

    def draws(self) -> List[Tile]:
        assert self._cpp_obj is not None
        return [Tile(t) for t in self._cpp_obj.draw_history()]  # type: ignore

    def action_mask(self, dtype=np.float32) -> np.ndarray:
        assert self._cpp_obj is not None
        return np.array(self._cpp_obj.action_mask(), dtype=dtype)  # type: ignore

    def kyotaku(self) -> int:
        assert self._cpp_obj is not None
        return self._cpp_obj.kyotaku()

    def honba(self) -> int:
        assert self._cpp_obj is not None
        return self._cpp_obj.honba()

    def tens(self) -> List[int]:
        assert self._cpp_obj is not None
        return self._cpp_obj.tens()

    def round(self) -> int:
        # 東一局:0, ..., 南四局:7, ...
        assert self._cpp_obj is not None
        return self._cpp_obj.round()

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()  # type: ignore

    def to_proto(self) -> mjxproto.Observation:
        assert self._cpp_obj is not None
        return json_format.Parse(self.to_json(), mjxproto.Observation())

    def save_svg(self, filename: str, view_idx: Optional[int] = None) -> None:
        assert filename.endswith(".svg")
        assert view_idx is None or 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)

    def show_svg(self, view_idx: Optional[int] = None) -> None:
        assert view_idx is None or 0 <= view_idx < 4

        observation = self.to_proto()
        show_svg(observation, target_idx=view_idx)

    def to_features(self, feature_name: str):
        assert feature_name in ("mjx-small-v0", "han22-v0", "mjx-large-v0")
        assert self._cpp_obj is not None

        if feature_name == "mjx-large-v0":
            feature = np.array(self.MjxLargeV0.produce(self))
            return feature

        # TODO: use ndarray in C++ side
        if feature_name == "han22-v0":
            feature = np.array(self._cpp_obj.to_features_2d(feature_name), dtype=np.bool8)  # type: ignore
            return feature
        feature = np.array(self._cpp_obj.to_features_2d(feature_name), dtype=np.int32)  # type: ignore
        return feature

    @staticmethod
    def add_legal_actions(obs_json: str) -> str:
        assert len(Observation(obs_json).legal_actions()) == 0, "Legal actions are alredy set."
        return _mjx.Observation.add_legal_actions(obs_json)

    @classmethod
    def from_proto(cls, proto: mjxproto.Observation) -> Observation:
        return Observation(json_format.MessageToJson(proto))

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> Observation:
        obs = cls()
        obs._cpp_obj = cpp_obj
        return obs

    class MjxLargeV0:
        @classmethod
        def produce(cls, obj: Observation) -> List[List[int]]:
            feature = []

            for func in [
                cls.current_hand,
                cls.target_tile,
                cls.under_riichis,
                cls.discarded_tiles,
                cls.discarded_from_hand,
                cls.opened_tiles,
                cls.dealer,
                cls.doras,
                cls.shanten,
                cls.effective_discards,
                cls.effective_draws,
                cls.ignored_tiles,
                cls.kyotaku,
                cls.rankings,
                cls.round,
                cls.honba,
                cls.dora_num_of_target,
                cls.dora_num_in_hand,
            ]:
                feature.extend(func(obj))

            return feature

        @staticmethod
        def current_hand(obs: Observation) -> List[int]:
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
        def target_tile(obs: Observation) -> List[int]:
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
                    return obs.draws()[-1]
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
        def under_riichis(obs: Observation) -> List[int]:
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
        def discarded_tiles(obs: Observation) -> List[int]:
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
        def discarded_from_hand(obs: Observation) -> List[int]:
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
        def opened_tiles(obs: Observation) -> List[int]:
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
        def dealer(obs: Observation) -> List[int]:
            feat = [[0] * 34 for _ in range(4)]
            feat[(obs.dealer() - obs.who() + 4) % 4] = [1] * 34
            return feat

        @staticmethod
        def doras(obs: Observation) -> List[int]:
            dora_cnt = [0] * 34
            for dora in obs.doras():
                dora_cnt[dora] += 1

            feat = [[0] * 34 for _ in range(4)]
            for dora in range(34):
                for i in range(dora_cnt[dora]):
                    feat[i][dora] = 1
            return feat

        @staticmethod
        def shanten(obs: Observation) -> List[int]:
            # feat[i] := shanten数が i以上か
            feat = [[0] * 34 for _ in range(7)]
            for i in range(obs.curr_hand().shanten_number() + 1):
                feat[i] = [1] * 34
            return feat

        @staticmethod
        def effective_discards(obs: Observation) -> List[int]:
            hand = obs.curr_hand()
            feat = [0] * 34
            tile_types = hand.effective_discard_types()
            for tt in tile_types:
                feat[tt] = 1
            return [feat]

        @staticmethod
        def effective_draws(obs: Observation) -> List[int]:
            hand = obs.curr_hand()
            feat = [0] * 34
            tile_types = hand.effective_draw_types()
            for tt in tile_types:
                feat[tt] = 1
            return [feat]

        @staticmethod
        def ignored_tiles(obs: Observation) -> List[int]:
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
        def kyotaku(obs: Observation) -> List[int]:
            # feat[i] := 供託がi+1 本以上あるか
            feat = [[0] * 34 for _ in range(5)]
            for i in range(min(obs.kyotaku(), 5)):
                feat[i] = [1] * 34
            return feat

        @staticmethod
        def rankings(obs: Observation) -> List[int]:
            # feat[q*3 + i] := player q の現在の着順(0,1,2,3) がi+1 以上か
            feat = [[0] * 34 for _ in range(3 * 4)]
            for p, r in enumerate(utils.rankings(obs.tens())):
                q = (p + 4 - obs.who()) % 4
                for i in range(r):
                    feat[q * 3 + i] = [1] * 34
            return feat

        @staticmethod
        def round(obs: Observation) -> List[int]:
            # 東一局:0, ..., 南四局:7
            # feat[i] := 現在のroundが i+1以上か
            feat = [[0] * 34 for _ in range(7)]
            for i in range(min(obs.round(), 7)):
                feat[i] = [1] * 34
            return feat

        @staticmethod
        def honba(obs: Observation) -> List[int]:
            # feat[i] := 現在の本場がi+1 以上か
            feat = [[0] * 34 for _ in range(5)]
            for i in range(min(obs.honba(), 5)):
                feat[i] = [1] * 34
            return feat

        @staticmethod
        def dora_num_of_target(obs: Observation) -> List[int]:
            def _target_tile() -> Optional[Tile]:
                events = obs.events()
                if len(events) == 0:
                    return None

                last_event = events[-1]
                if last_event.type() in [EventType.DISCARD, EventType.TSUMOGIRI]:
                    return last_event.tile()

                elif last_event.type() == EventType.DRAW:
                    return obs.draws()[-1]
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
        def dora_num_in_hand(obs: Observation) -> List[int]:
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
