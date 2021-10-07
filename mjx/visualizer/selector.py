from typing import Optional

import inquirer

from mjx.action import Action
from mjx.visualizer.converter import action_type_en, action_type_ja, get_tile_char
from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable
from mjxproto import Observation
from mjxproto.mjx_pb2 import ActionType


class Selector:
    @classmethod
    def select_from_MahjongTable(
        cls, table: MahjongTable, unicode: bool = False, ja: int = 0
    ) -> Optional[Action]:
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        table: MahjongTable
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        board_visualizer = GameBoardVisualizer(GameVisualConfig())
        board_visualizer.print(table)

        assert table.legal_actions != []

        legal_actions_proto = []
        for act in table.legal_actions:
            legal_actions_proto.append(act.to_proto())

        if legal_actions_proto[0].type == ActionType.ACTION_TYPE_DUMMY:
            return None

        choice = [
            str(i)
            + ":"
            + (action_type_en[action.type] if ja == 0 else action_type_ja[action.type])
            + "-"
            + get_tile_char(action.tile, unicode)
            for i, action in enumerate(legal_actions_proto)
        ]
        questions = [
            inquirer.List(
                "action",
                message=["Select your action", "選択肢を選んでください"][ja],
                choices=choice,
            ),
        ]
        answers = inquirer.prompt(questions)

        assert answers is not None

        idx = int(answers["action"].split(":")[0])
        selected = table.legal_actions[idx]

        return Action.select_from(
            idx=selected.to_idx(),
            legal_actions=table.legal_actions,
        )

    @classmethod
    def select_from_proto(
        cls,
        proto_data: Observation,
        unicode: bool = False,
        ja: int = 0,
    ) -> Optional[Action]:
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        proto_data: Observation proto
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        assert isinstance(proto_data, Observation)
        return cls.select_from_MahjongTable(
            MahjongTable.from_proto(proto_data), unicode=unicode, ja=ja
        )
