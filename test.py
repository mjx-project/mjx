import mjx.env
import random


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


if __name__ == '__main__':
    test_SingleAgentEnv()
    test_RLlibMahjongEnv()
