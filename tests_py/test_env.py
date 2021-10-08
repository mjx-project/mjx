import random

import _mjx

import mjx.agent
import mjx.env


def test_EnvRunner():
    agents = {
        "player_0": _mjx.RandomDebugAgent(),
        "player_1": _mjx.RandomDebugAgent(),
        "player_2": _mjx.RandomDebugAgent(),
        "player_3": _mjx.RandomDebugAgent(),
    }
    seed_generator = _mjx.RandomSeedGenerator(["player_0", "player_1", "player_2", "player_3"])
    runner = _mjx.EnvRunner(agents, seed_generator, 100, 4, 10, None, None)


def test_MjxEnv():
    random_agent = mjx.agent.RandomDebugAgent()

    random.seed(1234)
    env = mjx.env.MjxEnv()
    obs_dict = env.reset(1234)

    assert len(obs_dict) == 1
    assert "player_2" in obs_dict
    assert env.state.to_proto().hidden_state.wall[:5] == [24, 3, 87, 124, 97]

    while not env.done():
        action_dict = {}
        for agent, obs in obs_dict.items():
            action_dict[agent] = random_agent.act(obs)
        obs_dict = env.step(action_dict)
    rewards = env.rewards()

    assert len(rewards) == 4
    assert rewards["player_0"] == 90
    assert rewards["player_1"] == 0
    assert rewards["player_2"] == 45
    assert rewards["player_3"] == -135

    # test specifing daler order
    obs_dict = env.reset(1234, ["player_3", "player_1", "player_2", "player_0"])
    assert len(obs_dict) == 1
    assert "player_3" in obs_dict

    obs_dict = env.reset(1234)
    assert len(obs_dict) == 1
    assert "player_2" in obs_dict
