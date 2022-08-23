import random
import mjx


class RandomAgent(mjx.Agent):
    def __init__(self):
        super().__init__()

    # When you use neural network models
    # you may want to infer actions by batch
    def act_batch(self, observations):
        return [random.choice(obs.legal_actions()) for obs in observations]


agent = RandomAgent()
# act_batch is called instead of act
agent.serve("127.0.0.1:8080", batch_size=16)
