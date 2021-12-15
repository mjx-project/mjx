import glob
import os

from mjx.visualizer.svg import MahjongTable, save_svg


def test_svg():
    obs_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources/observation/*.json"
    )
    files = glob.glob(obs_files)
    for filename in files:
        for i in range(10):
            proto_data_list = MahjongTable.load_proto_data(filename)
            proto_data = proto_data_list[i]
            svgfile = f"obs_{i}.svg"
            save_svg(proto_data, svgfile)
            os.remove(svgfile)

    obs_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/state/*.json")
    files = glob.glob(obs_files)
    for filename in files:
        for i in range(10):
            proto_data_list = MahjongTable.load_proto_data(filename)
            proto_data = proto_data_list[i]
            svgfile = f"state_{i}.svg"
            save_svg(proto_data, svgfile)
            os.remove(svgfile)
