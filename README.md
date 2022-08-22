[![ci](https://github.com/mjx-project/mjx/actions/workflows/ci.yml/badge.svg)](https://github.com/mjx-project/mjx/actions/workflows/ci.yml)

<!-- 
<p align="center">
<img src="icons/1500w/png/color1/1-1_p1500.png" alt="mjx" width="150"/>
</p>
-->
<!-- 
<p align="center"> 
<img src="icons/1500w/png/color1/3_p1500.png" alt="mjx" width="200"/>
</p>
-->
<p align="center"> 
<img src="icons/SVG/2-2_svg.svg" alt="mjx" width="300"/>
</p>

# Features


- **Blazingly fast.** :zap:
  - Mjx is 100x faster than existing OSS simulators. See [benchmark test]().
- **Tenhou compatible.** :mahjong:
  - The implemented rule is compatible with that of [Tenhou](https://tenhou.net/). See [compatibility tests]().
- **Gym-like API.** :robot:
  - Mjx supports [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) API, which provides [gym](https://github.com/openai/gym)-like API for multi-agent games.
- **Easy distributed computing.**
  - For large-scale reinforcement leraning and evaluation, Mjx supports distributed computing with [gRPC](https://github.com/grpc/grpc).
- **Beautiful visualization.**
  - Mjx supports game log visualization thanks to Tenhou platform.

# Install

```
$ pip install mjx
```

# Requirements

Mjx supports `Python3.7` or later in `Linux` and `macOS Intel` (10.15 or later).
Currently `Windows` and `macOS Apple Silicon` are NOT supported.
Contributions for supporting `Windows` and `macOS Apple Silicon` are more than welcome!

# Example

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

# Sever Usage

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

  # When you use neural network models
  # you may want to infer actions by batch
  def act_batch(self, observations):
    return [random.choice(obs.legal_actions()) 
            for obs in observations]


agent = RandomAgent()
# act_batch is called instead of act
agent.serve("127.0.0.1:8080", batch_size=8)
```

</td>
<td>

```py

import mjx

host="127.0.0.1:8080"

mjx.run({
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

# How to develop
We recommend you to develop Mjx inside a container.
Easiest way is open this repository from VsCode.
Feel free to mention to @sotetsuk if you have any questions.

# License

MIT
