import glob
import os

from mjx.visualizer.svg import *


def test_svg():
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for file in files:
        for i in range(10):
            make_svg(file, i)

    obs_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/state/*.json")
    files = glob.glob(obs_files)
    for file in files:
        make_svg(file, 0)
