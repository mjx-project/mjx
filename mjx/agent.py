import random
from typing import List

import _mjx  # type: ignore

import mjx


class Agent(_mjx.Agent):  # type: ignore
    def __init__(self) -> None:
        _mjx.Agent.__init__(self)  # type: ignore

    def act(self, observation: mjx.Observation) -> mjx.Action:
        raise NotImplementedError

    def act_batch(self, observations: List[mjx.Observation]) -> List[mjx.Action]:
        return [self.act(obs) for obs in observations]

    def serve(self, socket_address):
        raise NotImplementedError

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self.act(mjx.Observation._from_cpp_obj(observation))._cpp_obj

    def _act_batch(self, observations: List[_mjx.Observation]) -> List[_mjx.Action]:  # type: ignore
        actions: List[mjx.Action] = self.act_batch(
            [mjx.Observation._from_cpp_obj(obs) for obs in observations]
        )
        return [action._cpp_obj for action in actions]


class RandomAgent(Agent):  # type: ignore
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:  # type: ignore
        return random.choice(observation.legal_actions())


class RandomDebugAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._agent = _mjx.RandomDebugAgent()  # type: ignore

    def act(self, observation: mjx.Observation) -> mjx.Action:
        return mjx.Action._from_cpp_obj(self._act(observation._cpp_obj))

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self._agent._act(observation)
