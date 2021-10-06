from __future__ import annotations

from typing import List, Optional

import _mjx  # type: ignore


class Action:
    def __init__(self, action_json: Optional[str] = None) -> None:
        self._cpp_obj: Optional[_mjx.Action] = None  # type: ignore
        if action_json is None:
            return

        self._cpp_obj = _mjx.Action(action_json)  # type: ignore

    def to_json(self) -> str:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_json()

    def to_idx(self) -> int:
        assert self._cpp_obj is not None
        return self._cpp_obj.to_idx()

    @classmethod
    def select_from(cls, idx: int, legal_actions: List[Action]) -> Action:
        return _mjx.Action.select_from(idx, [a._cpp_obj for a in legal_actions])  # type: ignore

    @classmethod
    def _from_cpp_obj(cls, cpp_obj: _mjx.Action) -> Action:  # type: ignore
        action = cls()
        action._cpp_obj = cpp_obj
        return action
