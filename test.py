import mjx.env
import random
from pettingzoo.test import api_test


def test_SingleAgentEnv():
    random.seed(1234)
    env = mjx.env.SingleAgentEnv()
    env.seed(1234)
    obs = env.reset()
    done = False
    while not done:
        legal_actions = [i for i, b in enumerate(obs["action_mask"]) if b]
        action = random.choice(legal_actions)
        obs, reward, done, info = env.step(action)
    assert reward == 0


def test_RLlibMahjongEnv():
    random.seed(1234)
    env = mjx.env.RLlibMahjongEnv()
    env.seed(1234)
    obs_dict = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        action_dict = {}
        for agent, obs in obs_dict.items():
            legal_actions = [i for i, b in enumerate(obs["action_mask"]) if b]
            action_dict[agent] = random.choice(legal_actions)
        obs_dict, rewards, dones, info = env.step(action_dict)
    assert len(rewards) == 4
    assert rewards['player_0'] == 0
    assert rewards['player_1'] == 45
    assert rewards['player_2'] == 90
    assert rewards['player_3'] == -135


def test_PettingZooMahjongEnv():
    random.seed(1234)
    env = mjx.env.PettingZooMahjongEnv()
    env.seed(1234)
    env.reset()
    results = []
    for agent in env.agent_iter():
        observation, reward, done, info = env.last(True)
        results.append((agent, observation, reward, done, info))
        legal_actions = [i for i, b in enumerate(observation["action_mask"]) if b]
        action = random.choice(legal_actions)
        env.step(action if not done else None)

    # PettingZoo Test
    # https://github.com/mjx-project/mjx/pull/887
    # Last iteration before done
    agent, observation, reward, done, info = results[-5]
    assert agent == "player_2"
    assert observation["action_mask"][180] == 0
    assert reward == 0
    assert not done
    assert info == {}

    # After done
    agent, observation, reward, done, info = results[-4]
    assert agent == "player_1"
    assert observation["action_mask"][180] == 1
    assert reward == 45
    assert done
    assert info == {}

    agent, observation, reward, done, info = results[-3]
    assert agent == "player_3"
    assert observation["action_mask"][180] == 1
    assert reward == -135
    assert done
    assert info == {}

    agent, observation, reward, done, info = results[-2]
    assert agent == "player_0"
    assert observation["action_mask"][180] == 1
    assert reward == 0
    assert done
    assert info == {}

    agent, observation, reward, done, info = results[-1]
    assert agent == "player_2"
    assert observation["action_mask"][180] == 1
    assert reward == 90
    assert done
    assert info == {}

    # API test
    api_test(env, num_cycles=100000, verbose_progress=False)

if __name__ == '__main__':
    test_SingleAgentEnv()
    test_RLlibMahjongEnv()
    test_PettingZooMahjongEnv()
