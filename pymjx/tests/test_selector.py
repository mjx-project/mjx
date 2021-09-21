import glob
import os

from mjx.visualizer.selector import Selector


def test_selector():
    mode = "obs"
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for file in files:
        selector = Selector(file, mode, ja=0, unicode=True)
        selector.run()