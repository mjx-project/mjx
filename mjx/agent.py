import random

import _mjx  # type: ignore

import mjx


class Agent(_mjx.Agent):  # type: ignore
    def __init__(self) -> None:
        _mjx.Agent.__init__(self)  # type: ignore

    def act_impl(self, observation: mjx.Observation) -> mjx.Action:
        raise NotImplementedError

    def act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self.act_impl(mjx.Observation._from_cpp_obj(observation))._cpp_obj


class RandomAgent(Agent):  # type: ignore
    def __init__(self) -> None:
        super().__init__()

    def act_impl(self, observation: mjx.Observation) -> mjx.Action:  # type: ignore
        return random.choice(observation.legal_actions())
