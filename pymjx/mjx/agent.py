class Agent:
    def __init__(self, strategy: str):
        import mjx._mjx as _mjx

        self.agent = _mjx.Agent(strategy)

    def take_action(self, obs):
        return self.agent.take_action(obs)
