import glob
import os

from mjx.visualizer.selector import Selector
from mjx.visualizer.visualizer import MahjongTable
from mjxproto.mjx_pb2 import Observation

"""
このテストではカーソルキーを利用して選択する動作を行いますが、
pytestで実行するのが難しいため、わざとファイル名を変えて自動テストされないようにしています。

動作を確認するときは、通常のpythonファイルと同様に
python cli_selector_test.py
としてください。
"""


def select_test():
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for filename in files:
        for i in range(5):
            proto_data_list = MahjongTable.load_proto_data(filename)
            proto_data = MahjongTable.from_proto(proto_data_list[i])
            action = Selector.select_from_MahjongTable(proto_data)
            print(f"action:{action}")
            action = Selector.select_from_MahjongTable(proto_data, unicode=True, ja=1)
            print(f"action:{action}")

        for i in range(5):
            proto_data_list = MahjongTable.load_proto_data(filename)
            assert isinstance(proto_data_list[0], Observation)
            action = Selector.select_from_proto(proto_data_list[i])
            print(f"action:{action}")
            action = Selector.select_from_proto(proto_data_list[i], unicode=True, ja=1)
            print(f"action:{action}")


if __name__ == "__main__":
    select_test()
