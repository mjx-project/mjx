from __future__ import annotations

import _mjx
from typing import List


class Action:
    def __init__(self, action_json=None) -> None:
        self._cpp_obj = None
        if action_json is None:
            return

        self._cpp_obj = _mjx.Action(action_json)

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()

    def to_idx(self) -> int:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_idx()

    @classmethod
    def select_from(cls, idx: int, legal_actions: List[Action]) -> Action:
        return _mjx.Action.select_from(idx, [a._cpp_obj for a in legal_actions])

    @classmethod
    def _from_cpp_obj(cls, cpp_obj) -> Action:
        action = cls()
        action._cpp_obj = cpp_obj
        return action
