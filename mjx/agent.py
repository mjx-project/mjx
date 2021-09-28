import random

import _mjx  # type: ignore

import mjx


class RandomAgent(_mjx.Agent):  # type: ignore
    def __init__(self) -> None:
        _mjx.Agent.__init__(self)  # type: ignore

    def act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return random.choice(observation.legal_actions())
