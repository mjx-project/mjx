import json
from typing import List, Optional

import mjxproto
from google.protobuf import json_format
from mjx.visualizer.svg import save_svg

import mjx


class State:
    def __init__(self, cpp_obj):
        self._cpp_obj = cpp_obj

    def legal_actions(self) -> List[mjx.Action]:
        return [mjx.Action(cpp_obj) for cpp_obj in self._cpp_obj.legal_actions()]

    def to_json(self) -> str:
        return self._cpp_obj.to_json()

    def to_proto(self) -> mjxproto.State:
        json_data = self.to_json()
        return json_format.ParseDict(json.loads(json_data), mjxproto.State())

    def save_svg(self, filename: str, view_idx: int = 0):
        assert filename.endswith(".svg")
        assert 0 <= view_idx < 4

        observation = self.to_proto()
        save_svg(observation, filename, view_idx)
