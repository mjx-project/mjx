from __future__ import annotations

from typing import List, Optional

import _mjx  # type: ignore
import numpy as np  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.action import Action
from mjx.const import PlayerIdx, TileType
from mjx.event import Event
from mjx.feature import FeatureProducer
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
        assert feature_name in ("mjx-small-v0", "han22-v0", "col-34")
        assert self._cpp_obj is not None

        if feature_name == "col-34":
            keys = [
                "currentHand",
                "targetTile",
                "underRiichis",
                "discardedTiles",
                "discardedFromHand",
                "openedTiles",
                "shanten",
                "dealer",
                "doras",
                "effectiveDraws",
                "effectiveDiscards",
                "ignoredTiles",
                "kyotaku",
                "rankings",
                "round",
                "honba",
                "doraNumInHand",
                "doraNumOfTarget",
            ]
            feature = np.array(FeatureProducer.produce(self, keys))
            return feature

        # TODO: use ndarray in C++ side
        if feature_name == "han22-v0":
            feature = np.array(self._cpp_obj.to_features_2d(feature_name), dtype=np.bool8)  # type: ignore
            return feature
        feature = np.array(self._cpp_obj.to_features_2d(feature_name), dtype=np.int32)  # type: ignore
        return feature

    def to_features(self, keys: List[str]):
        return FeatureProducer.produce(self, keys)

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
