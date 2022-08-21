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
from mjx import MjxEnv
from mjx.agents import RandomAgent

agent = RandomAgent()
env = MjxEnv()
obs_dict = env.reset()
while not env.done():
    actions = {player_id: agent.act(obs)
            for player_id, obs in obs_dict.items()}
    obs_dict = env.step(actions)
returns = env.rewards()
```

