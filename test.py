import mjx.env
import random


def test_RLlibMahjongEnv():
    random.seed(1234)
    env = mjx.env.RLlibMahjongEnv()
    env.seed(1234)
    obs_dict = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        action_dict = {agent: random.choice(obs.legal_actions())
                       for agent, obs in obs_dict.items()}
        obs_dict, rewards, dones, info = env.step(action_dict)
    assert len(rewards) == 4
    assert rewards['player_0'] == 0
    assert rewards['player_1'] == 45
    assert rewards['player_2'] == 90
    assert rewards['player_3'] == -135


if __name__ == '__main__':
    test_RLlibMahjongEnv()
