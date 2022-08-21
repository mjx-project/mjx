import glob
import os

from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable


def test_visualizer():
    mode = "obs"
    show = True
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    board_visualizer = GameBoardVisualizer(GameVisualConfig())
    for file in files:
        proto_data_list = MahjongTable.load_proto_data(file)
        for proto_data in proto_data_list:
            mahjong_table = MahjongTable.from_proto(proto_data)
            assert isinstance(mahjong_table, MahjongTable)
            assert 4 == len(mahjong_table.players)
            if show:
                board_visualizer.print(mahjong_table)

    # board_visualizer = GameBoardVisualizer(GameVisualConfig(rich=True))
    # for file in files:
    #     proto_data_list = MahjongTable.load_proto_data(file)
    #     for proto_data in proto_data_list:
    #         mahjong_table = MahjongTable.from_proto(proto_data)
    #         assert isinstance(mahjong_table, MahjongTable)
    #         assert 4 == len(mahjong_table.players)
    #         if show:
    #             board_visualizer.print(mahjong_table)
