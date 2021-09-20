import importlib
import json
from google.protobuf import json_format

_mjx = importlib.import_module("mjx._mjx")


class Action:
    def __init__(self, action_json=None):
        self._cpp_obj = None
        if action_json is None:
            return

        self._cpp_obj = _mjx.Action(action_json)

    def to_json(self) -> str:
        return self._cpp_obj.to_json()

    def to_idx(self) -> str:
        return self._cpp_obj.to_idx()

    @classmethod
    def _from_cpp_obj(cls, cpp_obj):
        action = cls()
        action._cpp_obj = cpp_obj
        return action
