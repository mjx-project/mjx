import os
import sys

import inquirer

sys.path.append(os.path.dirname(__file__))
from converter import action_type_en, action_type_ja, get_tile_char
from mjxproto.mjx_pb2 import ActionType
from visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable


class Selector:
    def __init__(self, file, mode, ja: int = 0, unicode: bool = False):
        self.file = file
        self.mode = mode
        self.ja = ja
        self.uni = unicode

    def run(self):
        board_visualizer = GameBoardVisualizer(GameVisualConfig())

        mahjong_tables = MahjongTable.load_data(self.file, self.mode)
        mahjong_table = mahjong_tables[-1]

        board_visualizer.print(mahjong_table)
        if mahjong_table.legal_actions == []:
            mahjong_table.legal_actions = [
                [ActionType.ACTION_TYPE_CHI, 10],
                [ActionType.ACTION_TYPE_PON, 20],
                [ActionType.ACTION_TYPE_OPEN_KAN, 30],
            ]  # 空っぽの時でも結果を見たいので、ダミーを出しておく

        choice = [
            (action_type_en[actions[0]] if self.ja == 0 else action_type_ja[actions[0]])
            + "-"
            + get_tile_char(actions[1], self.uni)
            for actions in mahjong_table.legal_actions
        ]
        print(choice)

        questions = [
            inquirer.List(
                "action",
                message=["Select your action", "選択肢を選んでください"][self.ja],
                choices=choice,
            ),
        ]
        answers = inquirer.prompt(questions)
        print(answers)
        item = answers["action"].split("-")[0]
        action = [k for k, v in [action_type_en, action_type_ja][self.ja].items() if v == item][0]
        print(action)

        # return action
