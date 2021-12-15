from typing import List

import inquirer

from mjx.action import Action
from mjx.open import Open
from mjx.visualizer.converter import action_type_en, action_type_ja, get_tile_char
from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable
from mjxproto import Observation
from mjxproto.mjx_pb2 import ActionType


class Selector:
    @classmethod
    def select_from_MahjongTable(
        cls, table: MahjongTable, unicode: bool = False, rich: bool = False, ja: bool = False
    ) -> Action:
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        table: MahjongTable
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        language: int = 1 if ja else 0
        board_visualizer = GameBoardVisualizer(
            GameVisualConfig(uni=unicode, rich=rich, lang=language)
        )
        board_visualizer.print(table)

        assert len(table.legal_actions) != 0

        legal_actions_proto = []
        for act in table.legal_actions:
            legal_actions_proto.append(act.to_proto())

        if (
            legal_actions_proto[0].type == ActionType.ACTION_TYPE_DUMMY
            or len(table.legal_actions) == 1
        ):  # 選択肢がダミーだったり一つしかないときは、そのまま返す
            return table.legal_actions[0]

        choices = cls.make_choices(legal_actions_proto, unicode, ja)

        questions = [
            inquirer.List(
                "action",
                message=("行動を選んでください" if ja else "Select your action"),
                choices=choices,
            ),
        ]
        answers = inquirer.prompt(questions)
        assert answers is not None
        idx = int(answers["action"].split(":")[0])
        return table.legal_actions[idx]

    @classmethod
    def make_choice(cls, action, i, unicode, ja) -> str:
        if action.type == ActionType.ACTION_TYPE_NO:
            return (
                str(i) + ":" + (action_type_ja[action.type] if ja else action_type_en[action.type])
            )

        elif action.type in [
            ActionType.ACTION_TYPE_PON,
            ActionType.ACTION_TYPE_CHI,
            ActionType.ACTION_TYPE_CLOSED_KAN,
            ActionType.ACTION_TYPE_OPEN_KAN,
            ActionType.ACTION_TYPE_ADDED_KAN,
            ActionType.ACTION_TYPE_RON,
        ]:
            open_data = Open(action.open)
            open_tile_ids = [tile.id() for tile in open_data.tiles()]
            return (
                str(i)
                + ":"
                + (action_type_ja[action.type] if ja else action_type_en[action.type])
                + "-"
                + " ".join([get_tile_char(id, unicode) for id in open_tile_ids])
            )

        else:
            return (
                str(i)
                + ":"
                + (action_type_ja[action.type] if ja else action_type_en[action.type])
                + "-"
                + get_tile_char(action.tile, unicode)
            )

    @classmethod
    def make_choices(cls, legal_actions_proto, unicode, ja) -> List[str]:
        choices = []
        for i, action in enumerate(legal_actions_proto):
            choices.append(cls.make_choice(action, i, unicode, ja))

        return choices

    @classmethod
    def select_from_proto(
        cls,
        proto_data: Observation,
        unicode: bool = False,
        rich: bool = False,
        ja: bool = False,
    ) -> Action:
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        proto_data: Observation proto
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        assert isinstance(proto_data, Observation)
        return cls.select_from_MahjongTable(
            MahjongTable.from_proto(proto_data), unicode=unicode, rich=rich, ja=ja
        )
