import unittest
import glob

from mjx.visualizer.visualizer import GameVisualConfig, GameBoardVisualizer, MahjongTable


class VisualizerTest(unittest.TestCase):
    def test_visualizer(self):
        mode = "obs"
        show = False
        files = glob.glob("json/*.json")
        for file in files:
            game_data = MahjongTable.load_data(file, mode)
            self.assertEqual(list, type(game_data))
            self.assertEqual(MahjongTable, type(game_data[0]))
            self.assertEqual(4, len(game_data[0].players))

            if show:
                board_visualizer = GameBoardVisualizer(GameVisualConfig())
                board_visualizer.print(game_data[0])


if __name__ == "__main__":
    unittest.main()
