import mjxcore


def test_RLlibMahjongEnv():
    env = mjxcore.RLlibMahjongEnv()
    env.seed(1234)
    obs = env.reset()
    assert len(obs.keys()) == 1
    assert 'player_2' in obs.keys()
