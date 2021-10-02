import random

import _mjx

import mjx.agent
import mjx.env


def test_RandomAgent():
    agents = {
        "player_0": mjx.agent.RandomAgent(),
        "player_1": mjx.agent.RandomAgent(),
        "player_2": mjx.agent.RandomAgent(),
        "player_3": mjx.agent.RandomAgent(),
    }
    runner = _mjx.EnvRunner(agents, 0, 0, True)
    while True:
        state = runner.pop_state()
        if not state:
            break
        print(state)


def test_RandomDebugAgent():
    agents = {
        "player_0": mjx.agent.RandomDebugAgent(),
        "player_1": mjx.agent.RandomDebugAgent(),
        "player_2": mjx.agent.RandomDebugAgent(),
        "player_3": mjx.agent.RandomDebugAgent(),
    }
    runner = _mjx.EnvRunner(agents, 0, 0, 1)
    while True:
        state = runner.pop_state()
        if not state:
            break
        print(state)


def test_RuleBasedAgent():
    agents = {
        "player_0": mjx.agent.RuleBasedAgent(),
        "player_1": mjx.agent.RuleBasedAgent(),
        "player_2": mjx.agent.RuleBasedAgent(),
        "player_3": mjx.agent.RuleBasedAgent(),
    }
    runner = _mjx.EnvRunner(agents, 0, 0, 1)
    while True:
        state = runner.pop_state()
        if not state:
            break
        print(state)
