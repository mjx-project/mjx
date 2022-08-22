from __future__ import annotations

import itertools
from typing import List, Optional

import _mjx  # type: ignore
import numpy as np  # type: ignore
from google.protobuf import json_format

import mjxproto
from mjx.action import Action
from mjx.const import ActionType, EventType, PlayerIdx, TileType
from mjx.event import Event
from mjx.hand import Hand
from mjx.tile import Tile
from mjx.visualizer.svg import save_svg, show_svg, to_svg
from mjx.visualizer.visualizer import MahjongTable


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
        assert feature_name in ("mjx-small-v0", "han22-v0")
        if feature_name == "han22-v0":
            feature = self._get_han22_features()
            return feature

        assert self._cpp_obj is not None
        # TODO: use ndarray in C++ side
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

    def _get_han22_features(self) -> np.ndarray:
        feature = np.full((93, 34), False, dtype=bool)
        proto = self.to_proto()
        mj_table = MahjongTable.from_proto(proto)

        closed_tiles_ = list(
            filter(
                lambda tile_unit: tile_unit.tile_unit_type == EventType.DRAW,
                mj_table.players[self.who()].tile_units,
            )
        )[0]
        closed_tiles_id = [tile.id() for tile in closed_tiles_.tiles]
        closed_tiles_type = [id // 4 for id in closed_tiles_id]

        for tiletype in range(34):  # tiletype: 0~33

            # 0-5
            in_hand = closed_tiles_type.count(tiletype)
            feature[0][tiletype] = in_hand > 0
            feature[1][tiletype] = in_hand > 1
            feature[2][tiletype] = in_hand > 2
            feature[3][tiletype] = in_hand == 4

            for event in proto.public_observation.events:
                if (
                    (
                        event.type == mjxproto.EVENT_TYPE_DISCARD
                        or event.type == mjxproto.EVENT_TYPE_TSUMOGIRI
                    )
                    and event.who == proto.who
                    and event.tile // 4 == tiletype
                ):
                    feature[4][tiletype] = True
                    break

            feature[5][tiletype] = tiletype in [4, 13, 22] and (
                tiletype * 34 in closed_tiles_id
                or tiletype * 34 + 1 in closed_tiles_id
                or tiletype * 34 + 2 in closed_tiles_id
                or tiletype * 34 + 3 in closed_tiles_id
            )

            # 6-29
            for j in range(4):
                player_id = (self.who() + j) % 4
                _calling_of_player_j = self._calling_of_player_i(tiletype, player_id, mj_table)
                for k in range(6):
                    feature[6 + j * 6 + k][tiletype] = _calling_of_player_j[k]

            # 30-69
            for j in range(4):
                player_id = (self.who() + j) % 4
                _discarded_tiles_from_player_j = self._discarded_tiles_from_player_i(
                    tiletype, player_id, mj_table
                )
                for k in range(10):
                    feature[30 + j * 10 + k][tiletype] = _discarded_tiles_from_player_j[k]

            # 70-79
            for j in range(len(mj_table.doras)):
                feature[70 + j][tiletype] = (mj_table.doras[j]) // 4 - 1 == tiletype
                feature[74 + j][tiletype] = (mj_table.doras[j]) // 4 == tiletype
            feature[78][tiletype] = [27, 28, 29, 30][
                (mj_table.round - 1) // 4
            ] == tiletype  # 27=EW,28=SW,roundは1,2,3,..
            feature[79][tiletype] = [27, 28, 29, 30][mj_table.players[0].wind] == tiletype

            # 80
            if mj_table.latest_tile is not None:
                feature[80][tiletype] = mj_table.latest_tile // 4 == tiletype

            # 82-84
            if tiletype <= 26:
                if tiletype % 9 < 7:
                    feature[81][tiletype] = (
                        tiletype + 1 in closed_tiles_type and tiletype + 2 in closed_tiles_type
                    )
                if 0 < tiletype % 9 < 8:
                    feature[82][tiletype] = (
                        tiletype - 1 in closed_tiles_type and tiletype + 1 in closed_tiles_type
                    )
                if 1 < tiletype % 9:
                    feature[83][tiletype] = (
                        tiletype - 2 in closed_tiles_type and tiletype - 1 in closed_tiles_type
                    )

            # 85-92,81
            _information_for_available_actions = self._information_for_available_actions(
                tiletype, proto
            )
            for j in range(len(_information_for_available_actions) - 1):
                feature[85 + j][tiletype] = _information_for_available_actions[j]
            feature[92][tiletype] = feature[92][tiletype] and feature[0][tiletype]
            feature[81][tiletype] = _information_for_available_actions[8]

        return feature

    def _calling_of_player_i(self, tile_type: int, player_id: int, mj_table: MahjongTable):
        feature = [False] * 6
        tile_units = mj_table.players[player_id].tile_units
        open_tile_units = list(
            filter(
                lambda tile_unit: tile_unit.tile_unit_type != EventType.DRAW
                and tile_unit.tile_unit_type != EventType.DISCARD,
                tile_units,
            )
        )
        open_tiles_id_ = [
            [tile.id() for tile in open_tile_unit.tiles] for open_tile_unit in open_tile_units
        ]
        open_tiles_id = list(itertools.chain.from_iterable(open_tiles_id_))
        open_tiles_type = [id // 4 for id in open_tiles_id]

        in_furo = open_tiles_type.count(tile_type)
        feature[0] = in_furo > 0
        feature[1] = in_furo > 1
        feature[2] = in_furo > 2
        feature[3] = in_furo == 4

        stolen_tiles = [open_tile_unit.tiles[0].id() // 4 for open_tile_unit in open_tile_units]
        feature[4] = tile_type in stolen_tiles

        feature[5] = tile_type in [4, 13, 22] and (
            tile_type * 34 in open_tiles_id
            or tile_type * 34 + 1 in open_tiles_id
            or tile_type * 34 + 2 in open_tiles_id
            or tile_type * 34 + 3 in open_tiles_id
        )

        return feature

    def _discarded_tiles_from_player_i(
        self, tile_type: int, player_id: int, mj_table: MahjongTable
    ):
        feature = [False] * 10
        tile_units = mj_table.players[player_id].tile_units
        discard_tile_unit = list(
            filter(
                lambda tile_unit: tile_unit.tile_unit_type == EventType.DISCARD,
                tile_units,
            )
        )[0]
        discard_tiles_id = [tile.id() for tile in discard_tile_unit.tiles]
        discard_tiles_type = [id // 4 for id in discard_tiles_id]

        tile_in_discard = list(
            filter(
                lambda tile: tile.id() // 4 == tile_type,
                discard_tile_unit.tiles,
            )
        )

        in_discard = discard_tiles_type.count(tile_type)
        feature[0] = in_discard > 0
        feature[1] = in_discard > 1
        feature[2] = in_discard > 2
        feature[3] = in_discard == 4

        for j in range(len(tile_in_discard)):
            feature[4 + j] = not tile_in_discard[j].is_tsumogiri
            if tile_in_discard[j].with_riichi:
                feature[9] = True

        feature[8] = tile_type in [4, 13, 22] and (
            tile_type * 34 in discard_tiles_id
            or tile_type * 34 + 1 in discard_tiles_id
            or tile_type * 34 + 2 in discard_tiles_id
            or tile_type * 34 + 3 in discard_tiles_id
        )

        return feature

    def _information_for_available_actions(self, tile_type: int, proto):
        feature = [False] * (8 + 1)  # 最後の1は81番用
        obs = Observation.from_proto(proto)
        try:
            legal_actions = Observation(obs.add_legal_actions(obs.to_json())).legal_actions()
        except AssertionError:
            legal_actions = obs.legal_actions()

        for action in legal_actions:
            if (
                action.type() == ActionType.PON
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[0] = True
            if (
                action.type() == ActionType.CLOSED_KAN
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[1] = True
            if (
                action.type() == ActionType.OPEN_KAN
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[2] = True
            if (
                action.type() == ActionType.ADDED_KAN
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[3] = True
            if (
                action.type() == ActionType.RIICHI
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[4] = True
            if (
                action.type() == ActionType.RON
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[5] = True
            if (
                action.type() == ActionType.TSUMO
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[6] = True
            if action.type() == ActionType.ABORTIVE_DRAW_NINE_TERMINALS:
                feature[7] = True
            if (
                action.type() == ActionType.DISCARD
                and action.tile() is not None
                and action.tile().type() == tile_type
            ):
                feature[8] = True
        return feature
