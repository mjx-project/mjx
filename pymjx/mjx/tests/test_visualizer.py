import unittest
from mjx.visualizer.visualizer import GameVisualConfig, GameBoardVisualizer, MahjongTable


class VisualizerTest(unittest.TestCase):
    def test_visualizer(self):
        path = "observations.json"
        mode = "obs"
        print = False

        game_data = MahjongTable.load_data(path, mode)

        self.assertEqual(list, type(game_data))
        self.assertEqual(MahjongTable, type(game_data[0]))
        self.assertEqual(4, len(game_data[0].players))

        if print:
            board_visualizer = GameBoardVisualizer(GameVisualConfig())
            board_visualizer.print(game_data)


if __name__ == "__main__":
    unittest.main()
