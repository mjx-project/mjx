import glob
import os
from mjx.visualizer.svg import *


def test_svg():
    mode = "obs"
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for file in files:
        make_svg(file, mode, 0)

    mode = "sta"
    obs_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/state/*.json")
    files = glob.glob(obs_files)
    for file in files:
        make_svg(file, mode, 0)