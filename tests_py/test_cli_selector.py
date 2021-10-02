import glob
import os

from mjx.visualizer.selector import Selector
from mjx.visualizer.visualizer import MahjongTable

"""
このテストではカーソルキーを利用して選択する動作を行うが、
pytestで実行するのが難しいためわざと自動テストされないようにしている
"""


def select_test():
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for filename in files:
        proto_data_list = MahjongTable.load_proto_data(filename)
        proto_data = MahjongTable.from_proto(proto_data_list[0])
        answer = Selector.select_from_MahjongTable(proto_data)
        print(f"You selected '{answer}'.")

    obs_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/state/*.json")
    files = glob.glob(obs_files)
    for filename in files:
        proto_data_list = MahjongTable.load_proto_data(filename)
        proto_data = MahjongTable.from_proto(proto_data_list[0])
        answer = Selector.select_from_MahjongTable(proto_data)
        print(f"You selected '{answer}'.")


if __name__ == "__main__":
    select_test()
