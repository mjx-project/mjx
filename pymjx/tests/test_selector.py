import glob
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mjx.visualizer.selector import Selector


def selector():
    mode = "obs"
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for file in files:
        selector = Selector(file, mode, ja=0, unicode=True)
        selector.run()


if __name__ == "__main__":
    selector()
