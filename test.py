import mjx._mjx as m


def test_RLlibMahjongEnv():
    env = m.RLlibMahjongEnv()
    env.seed(1234)
    obs = env.reset()
    assert len(obs.keys()) == 1
    assert 'player_2' in obs.keys()