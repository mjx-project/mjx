import numpy as np
import torch
import torch.nn.functional as F


class Agent:
    def __init__(self, strategy: str):
        import mjx._mjx as _mjx

        if strategy in ["rule_based"]:
            self.agent = _mjx.Agent(strategy)
        if strategy in ["small_v0"]:
            self.agent = AgentSmallV0()

    def take_action(self, obs):
        return self.agent.take_action(obs)


class AgentSmallV0:
    class Net(torch.nn.Module):
        def __init__(self, num_in, num_out, num_layers, num_units):
            super().__init__()
            self.linears = torch.nn.ModuleList([torch.nn.Linear(num_in, num_units)])
            self.linears.extend([torch.nn.Linear(num_units, num_units) for _ in range(num_layers)])
            self.fc = torch.nn.Linear(num_units, num_out)

        def forward(self, x):
            for linear in self.linears:
                x = linear(x)
                x = F.relu(x)
            x = self.fc(x)
            output = F.log_softmax(x, dim=1)
            return output

    def __init__(self):
        # load model
        num_layers = 2
        num_units = 2048
        lr = 1e-3
        name = f"small_v0_not_jit-lr={lr}-num_layers={num_layers}-num_units={num_units}"
        MODEL_DIR = (
            "/Users/habarakeigo/ghq/github.com/mjx-project/mjx/pymjx/mjx/resources"  # FIXME
        )
        self.net = self.Net(34 * 10, 180, num_layers, num_units)
        self.net.load_state_dict(torch.load(MODEL_DIR + "/" + name + ".pt"))

    def take_action(self, obs):
        import mjx._mjx as _mjx

        # filter [0, 180) and select argmax
        legal_actions = obs.legal_actions()

        if len(legal_actions) == 1:
            return legal_actions[0]

        feature = obs.to_feature("small_v0")
        output = self.net.forward(torch.FloatTensor([feature]))
        mask = [True] * 180
        for a in legal_actions:
            mask[a.to_idx()] = False

        selected = np.ma.masked_array(output.detach().numpy().copy(), mask).argmax()

        return _mjx.Action(selected, legal_actions)
