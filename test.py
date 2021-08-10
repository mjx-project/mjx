import mjx.env
import mjx.agent
import random

def test_AgentRuleBased():
    env = mjx.env.RLlibMahjongEnv()
    rule_based = mjx.agent.Agent("rule_based")

    def run():
        obs_dict = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            action_dict = {agent: rule_based.take_action(obs) if agent == "player_0" else random.choice(obs.legal_actions())
                           for agent, obs in obs_dict.items()}
            obs_dict, rewards, dones, info = env.step(action_dict)
        assert len(rewards) == 4
        return rewards

    wins = {
        'player_0': 0,
        'player_1': 0,
        'player_2': 0,
        'player_3': 0,
    }

    for t in range(100):
        rewards = run()
        for agent, point in rewards.items():
            wins[agent] += point

    # rule_based agent must take first place
    assert(wins['player_0'] == 90 * 100)


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
    test_AgentRuleBased()
