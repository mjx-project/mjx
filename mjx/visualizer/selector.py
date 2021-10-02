from typing import Union

import inquirer

import mjxproto
from mjx.visualizer.converter import action_type_en, action_type_ja, get_tile_char
from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable
from mjxproto.mjx_pb2 import ActionType


class Selector:
    def __init__(self):
        pass

    @classmethod
    def select_from_MahjongTable(cls, table: MahjongTable, unicode: bool = False, ja: int = 0):
        board_visualizer = GameBoardVisualizer(GameVisualConfig())
        board_visualizer.print(table)

        if table.legal_actions == []:
            table.legal_actions = [
                (ActionType.ACTION_TYPE_CHI, 10),
                (ActionType.ACTION_TYPE_PON, 20),
                (ActionType.ACTION_TYPE_OPEN_KAN, 30),
            ]  # 空っぽの時でも結果を見たいので、ダミーを出しておく

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

        if answers == None:
            print("Incorrect choice was made.")
            return ActionType.ACTION_TYPE_DUMMY

        item = answers["action"].split("-")[0]
        action = [k for k, v in [action_type_en, action_type_ja][ja].items() if v == item][0]
        return action

    @classmethod
    def select_from_proto(
        cls,
        proto_data: Union[mjxproto.Observation, mjxproto.State],
        unicode: bool = False,
        ja: int = 0,
    ):
        cls.select_from_MahjongTable(MahjongTable.from_proto(proto_data), unicode=unicode, ja=ja)
