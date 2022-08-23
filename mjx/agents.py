import random
from typing import Dict, List

import _mjx  # type: ignore

from mjx.action import Action
from mjx.const import ActionType
from mjx.env import MjxEnv
from mjx.observation import Observation
from mjx.visualizer.selector import Selector


class Agent(_mjx.Agent):  # type: ignore
    def __init__(self) -> None:
        _mjx.Agent.__init__(self)  # type: ignore

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    def act_batch(self, observations: List[Observation]) -> List[Action]:
        return [self.act(obs) for obs in observations]

    def serve(
        self,
        socket_address: str,
        batch_size: int = 64,
        wait_limit_ms: int = 0,
        sleep_ms: int = 0,
    ):
        _mjx.AgentServer(self, socket_address, batch_size, wait_limit_ms, sleep_ms)  # type: ignore

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self.act(Observation._from_cpp_obj(observation))._cpp_obj

    def _act_batch(self, observations: List[_mjx.Observation]) -> List[_mjx.Action]:  # type: ignore
        actions: List[Action] = self.act_batch(
            [Observation._from_cpp_obj(obs) for obs in observations]
        )
        return [action._cpp_obj for action in actions]


class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        return random.choice(observation.legal_actions())


class TsumogiriAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        for action in legal_actions:
            if action.type() in [ActionType.TSUMOGIRI, ActionType.PASS]:
                return action
        assert False


class ShantenAgent(Agent):
    """A rule-based agent, which plays just to reduce the shanten-number.
    The logic is basically intended to reproduce Mjai's ShantenPlayer.

    - Mjai https://github.com/gimite/mjai
    """

    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # if it can declare riichi, just declar riichi
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        # if it can apply chi/pon/open-kan, choose randomly
        steal_actions = [
            a
            for a in legal_actions
            if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            return random.choice(steal_actions)

        # if it can apply closed-kan/added-kan, choose randomly
        kan_actions = [
            a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN, ActionType.ADDED_KAN]
        ]
        if len(kan_actions) >= 1:
            return random.choice(kan_actions)

        # discard an effective tile randomly
        legal_discards = [
            a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
        ]
        effective_discard_types = observation.curr_hand().effective_discard_types()
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]
        if len(effective_discards) > 0:
            return random.choice(effective_discards)

        # if no effective tile exists, discard randomly
        return random.choice(legal_discards)


class RandomDebugAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._agent = _mjx.RandomDebugAgent()  # type: ignore

    def act(self, observation: Observation) -> Action:
        return Action._from_cpp_obj(self._act(observation._cpp_obj))

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self._agent._act(observation)


class RuleBasedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._agent = _mjx.RuleBasedAgent()  # type: ignore

    def act(self, observation: Observation) -> Action:
        return Action._from_cpp_obj(self._act(observation._cpp_obj))

    def _act(self, observation: _mjx.Observation) -> _mjx.Action:  # type: ignore
        return self._agent._act(observation)


class HumanControlAgent(Agent):  # type: ignore
    def __init__(self, unicode: bool = False, rich: bool = False, ja: bool = False) -> None:
        super().__init__()
        self.unicode: bool = unicode
        self.ja: bool = ja
        self.rich: bool = rich

    def act(self, observation: Observation) -> Action:  # type: ignore
        return Selector.select_from_proto(
            observation.to_proto(), unicode=self.unicode, rich=self.rich, ja=self.ja
        )


def validate_agent(agent: Agent, n_games=1, use_batch=False):
    env = MjxEnv()
    for i in range(n_games):
        obs_dict: Dict[str, Observation] = env.reset()
        while not env.done():
            action_dict: Dict[str, Action] = {}
            for player_id, obs in obs_dict.items():
                action = agent.act_batch([obs])[0] if use_batch else agent.act(obs)
                assert action in obs.legal_actions()
                action_dict[player_id] = action
            obs_dict = env.step(action_dict)
