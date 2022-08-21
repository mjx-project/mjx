from mjx.agents import (
    RandomAgent,
    RandomDebugAgent,
    RuleBasedAgent,
    ShantenAgent,
    TsumogiriAgent,
    validate_agent,
)


def test_RandomAgent():
    agent = RandomAgent()
    validate_agent(agent, n_games=1, use_batch=True)
    validate_agent(agent, n_games=1, use_batch=False)


def test_RandomDebugAgent():
    agent = RandomDebugAgent()
    validate_agent(agent, n_games=1, use_batch=True)
    validate_agent(agent, n_games=1, use_batch=False)


def test_ShantenAgent():
    agent = ShantenAgent()
    validate_agent(agent, n_games=1, use_batch=True)
    validate_agent(agent, n_games=1, use_batch=False)


def test_RuleBasedAgent():
    agent = RuleBasedAgent()
    validate_agent(agent, n_games=1, use_batch=True)
    validate_agent(agent, n_games=1, use_batch=False)


def test_TsumogiriAgent():
    agent = TsumogiriAgent()
    validate_agent(agent, n_games=1, use_batch=True)
    validate_agent(agent, n_games=1, use_batch=False)
