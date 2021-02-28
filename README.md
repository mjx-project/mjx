[![](https://github.com/sotetsuk/mahjong/workflows/build/badge.svg)](https://github.com/sotetsuk/mahjong/actions)

# Mjx

Mjx is a Mahjong environment for AI research and development.
It supports [Japanese (Riichi) Mahjong](https://en.wikipedia.org/wiki/Japanese_Mahjong), one of the most popular variants of Mahjong in the world.

## Quick start


<table>
<tr>
<td align="center"> Python </td>
<td align="center"> C++ </td>
</tr>
<tr>
<td valign="top">

```py

import mjx 


env = make_env(reward="TenhouPhoenixRoom")
env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = random_select(observation.possible_actions)
    env.step(action)
```

</td>
<td valign="top">

```c++
#include <mjx/mjx.h>

void main() {
  auto env = mjx::MakeEnv(reward="TenhouPhoenixRoom");
  env.Reset();
  for (auto agent: env.AgentIter()) {
    auto [observation, reward, done, info] = env.Last();
    auto action = RandomSelect(observation.possible_actions);
    env.Step(action);
  }
}
```

</td>
</tr>
</table>

## Features

- **Blazingly fast.** 100x faster than existing OSS simulators.
- **[Tenhou](https://tenhou.net/) (天鳳) compatible rule.** One of the most popular variant in the world.
- **Gym-like API supported.** Mjx supports [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) API, which provides [gym](https://github.com/openai/gym)-like API for multi-agent games.
