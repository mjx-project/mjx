class RLlibMahjongEnv:
    def __init__(self):
        import mjx._mjx as _mjx
        self.env = _mjx.RLlibMahjongEnv()

    def step(self, action_dict):
        return self.env.step(action_dict)

    def reset(self):
        return self.env.reset()

    def seed(self, seed: int):
        self.env.seed(seed)
