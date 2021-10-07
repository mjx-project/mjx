import inquirer

from mjx.action import Action
from mjx.visualizer.converter import action_type_en, action_type_ja, get_tile_char
from mjx.visualizer.open_utils import open_tile_ids
from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable
from mjxproto import Observation
from mjxproto.mjx_pb2 import ActionType


class Selector:
    @classmethod
    def select_from_MahjongTable(
        cls, table: MahjongTable, unicode: bool = False, ja: int = 0
    ) -> Action:
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        table: MahjongTable
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        board_visualizer = GameBoardVisualizer(GameVisualConfig())
        board_visualizer.print(table)

        assert len(table.legal_actions) != 0

        legal_actions_proto = []
        for act in table.legal_actions:
            legal_actions_proto.append(act.to_proto())

        if legal_actions_proto[0].type == ActionType.ACTION_TYPE_DUMMY:
            return table.legal_actions[0]

        choice = []
        for i, action in enumerate(legal_actions_proto):
            if action.type == ActionType.ACTION_TYPE_NO:
                choice.append(
                    str(i)
                    + ":"
                    + (action_type_en[action.type] if ja == 0 else action_type_ja[action.type])
                )
            elif action.type in [
                ActionType.ACTION_TYPE_PON,
                ActionType.ACTION_TYPE_CHI,
                ActionType.ACTION_TYPE_CLOSED_KAN,
                ActionType.ACTION_TYPE_OPEN_KAN,
                ActionType.ACTION_TYPE_ADDED_KAN,
                ActionType.ACTION_TYPE_RON,
            ]:
                choice.append(
                    str(i)
                    + ":"
                    + (action_type_en[action.type] if ja == 0 else action_type_ja[action.type])
                    + "-"
                    + " ".join([get_tile_char(id, unicode) for id in open_tile_ids(action.open)])
                )
            else:
                choice.append(
                    str(i)
                    + ":"
                    + (action_type_en[action.type] if ja == 0 else action_type_ja[action.type])
                    + "-"
                    + get_tile_char(action.tile, unicode)
                )

        questions = [
            inquirer.List(
                "action",
                message=["Select your action", "行動を選んでください"][ja],
                choices=choice,
            ),
        ]
        answers = inquirer.prompt(questions)
        assert answers is not None
        idx = int(answers["action"].split(":")[0])
        return table.legal_actions[idx]

    @classmethod
    def select_from_proto(
        cls,
        proto_data: Observation,
        unicode: bool = False,
        ja: int = 0,
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
            MahjongTable.from_proto(proto_data), unicode=unicode, ja=ja
        )
