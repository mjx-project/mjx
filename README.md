[![](https://github.com/sotetsuk/mahjong/workflows/build/badge.svg)](https://github.com/sotetsuk/mahjong/actions)

# Mjx

**注: このREADMEで書いてある内容はあくまで最終目標で、まだ未実装**

Mjx is a Mahjong environment for AI research and development.
It supports [Japanese (Riichi) Mahjong](https://en.wikipedia.org/wiki/Japanese_Mahjong), one of the most popular variants of Mahjong in the world.

## Features

- **Blazingly fast.** 
  - Mjx is 100x faster than existing OSS simulators. See [benchmark]().
- **Tenhou compatible.** 
  - The implemented rule is compatible with that of [Tenhou](https://tenhou.net/).
- **Gym-like API compatible.** 
  - Mjx supports [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) API, which provides [gym](https://github.com/openai/gym)-like API for multi-agent games.

## Quick start

Mjx supports Python and C++17.


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

## Install

We highly recommend to use docker to use mjx without environment dependencies.


### Docker

```sh
$ docker pull mjx-project/mjx:latest
```

### For Python

```sh
$ pip install mjx
```

### For C++


```$
$ make build
$ make test
```

## LICENSE
TBA