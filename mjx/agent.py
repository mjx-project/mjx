import _mjx

import mjx


class RandomDebugAgent:
    def __init__(self) -> None:
        self._agent = _mjx.RandomDebugAgent()

    def act(self, observation):
        return mjx.Action._from_cpp_obj(self._agent.act(observation._cpp_obj))
