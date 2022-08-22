[![ci](https://github.com/mjx-project/mjx/actions/workflows/ci.yml/badge.svg)](https://github.com/mjx-project/mjx/actions/workflows/ci.yml)

<p align="center">
<img src="icons/1500w/png/color1/1-1_p1500.png" alt="mjx" width="150"/>
<p align="center">
<img src="icons/1500w/png/color1/3_p1500.png" alt="mjx" width="75"/>
</p>

## Requirements

- `Ubuntu 20.04` or later
- `MacOS 10.15` or later <!-- <filesystem> requires macos-10.15 -->
- `Python >= 3.7` <!-- importlib requires 3.7 -->

## Example

```py
import mjx

agent = mjx.RandomAgent()
env = mjx.MjxEnv()
obs_dict = env.reset()
while not env.done():
    actions = {player_id: agent.act(obs)
            for player_id, obs in obs_dict.items()}
    obs_dict = env.step(actions)
returns = env.rewards()
```

## Sever Usage

<table>
<tr><th>Server</th><th>Client</th></tr>

<tr>
<td>

```py
import random
import mjx

class RandomAgent(mjx.Agent):
    def __init__(self):
        super().__init__()

    # When you use neural network models you may want to infer actions by batch
    def act_batch(self, observations):
        return [random.choice(observation.legal_actions()) for obs in observations]


agent = RandomAgent()
agent.serve("127.0.0.1:8080", batch_size=8)
```

</td>
<td>

```py

import mjx

host="127.0.0.1"

mjx.run(
    {
        "player_0": host,
        "player_1": host,
        "player_2": host,
        "player_3": host
    },
    num_games=1000,
    num_parallels=16
)
```

</td>
</tr>
</table>