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
    _mjx.EnvRunner.run(agents)


def test_RandomDebugAgent():
    agents = {
        "player_0": mjx.agent.RandomDebugAgent(),
        "player_1": mjx.agent.RandomDebugAgent(),
        "player_2": mjx.agent.RandomDebugAgent(),
        "player_3": mjx.agent.RandomDebugAgent(),
    }
    _mjx.EnvRunner.run(agents)
