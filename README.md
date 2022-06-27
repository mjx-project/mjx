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
from mjx.agent import RandomAgent

agent = RandomAgent()
env = MjxEnv()
player_id, observation = env.reset()
while not env.done():
    action = agent.act(observation)
    player_id, observation = env.step(action)
rewards = env.rewards()
```

## Visualization

Available at https://mjx-visualizer.an.r.appspot.com/
