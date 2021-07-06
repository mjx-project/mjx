import glob
import os

from mjx.visualizer.visualizer import GameBoardVisualizer, GameVisualConfig, MahjongTable


def test_visualizer():
    mode = "obs"
    show = False
    obs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_dir, recursive=True)
    for file in files:
        game_data = MahjongTable.load_data(file, mode)
        assert isinstance(game_data, list)
        assert isinstance(game_data[0], MahjongTable)
        assert 4 == len(game_data[0].players)

        if show:
            board_visualizer = GameBoardVisualizer(GameVisualConfig())
            board_visualizer.print(game_data[0])
