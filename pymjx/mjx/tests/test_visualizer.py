import glob

from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable


def test_visualizer():
    mode = "obs"
    show = False
    files = glob.glob("resources/observation/*.json")
    for file in files:
        game_data = MahjongTable.load_data(file, mode)
        assert list == type(game_data)
        assert MahjongTable == type(game_data[0])
        assert 4 == len(game_data[0].players)

        if show:
            board_visualizer = GameBoardVisualizer(GameVisualConfig())
            board_visualizer.print(game_data[0])
