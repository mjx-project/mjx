from typing import Optional, Union

import inquirer

from mjx.visualizer.converter import (
    action_type_en,
    action_type_ja,
    get_tile_char,
    to_char,
    to_unicode,
)
from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable
from mjxproto import Observation, State
from mjxproto.mjx_pb2 import ActionType


class Selector:
    @classmethod
    def select_from_MahjongTable(cls, table: MahjongTable, unicode: bool = False, ja: int = 0):
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        table: MahjongTable
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        board_visualizer = GameBoardVisualizer(GameVisualConfig())
        board_visualizer.print(table)

        if table.legal_actions == []:
            # 本来は return None
            # 今は空っぽの時でも結果を見たいので、ダミーを出しておく
            table.legal_actions = [
                (ActionType.ACTION_TYPE_CHI, 10),
                (ActionType.ACTION_TYPE_PON, 20),
                (ActionType.ACTION_TYPE_OPEN_KAN, 30),
            ]

        choice = [
            (action_type_en[actions[0]] if ja == 0 else action_type_ja[actions[0]])
            + "-"
            + get_tile_char(actions[1], unicode)
            for actions in table.legal_actions
        ]
        questions = [
            inquirer.List(
                "action",
                message=["Select your action", "選択肢を選んでください"][ja],
                choices=choice,
            ),
        ]
        answers = inquirer.prompt(questions)

        if answers is None:
            print("Incorrect choice was made.")
            return (ActionType.ACTION_TYPE_DUMMY, int(0))

        item = answers["action"].split("-")[0]
        id_str: str = answers["action"].split("-")[1]
        id: Optional[int] = None
        if unicode:
            for i, k in enumerate(to_unicode):
                if k == id_str:
                    id = i
                    break
        else:
            for i, k in enumerate(to_char):
                if k == id_str:
                    id = i
                    break

        action = [k for k, v in [action_type_en, action_type_ja][ja].items() if v == item][0]
        return (action, id)

    @classmethod
    def select_from_proto(
        cls,
        proto_data: Union[Observation, State],
        unicode: bool = False,
        ja: int = 0,
    ):
        """Make selector from State/Observation MahjongTable data.

        Args
        ----
        proto_data: State or observation proto
        unicode: bool
        ja: int (0-English,1-Japanese)
        """
        return cls.select_from_MahjongTable(
            MahjongTable.from_proto(proto_data), unicode=unicode, ja=ja
        )
