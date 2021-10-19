from mjx.visualizer.svg import make_svg
from mjx.visualizer.visualizer import MahjongTable


def svg_from_json(json_data: str, target_idx: int) -> str:
    proto_data = MahjongTable.json_to_proto(json_data=json_data)
    dwg = make_svg(proto_data, ".svg", target_idx)
    return '<?xml version="1.0" encoding="utf-8" ?>\n' + dwg.tostring()
