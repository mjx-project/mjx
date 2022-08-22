[![ci](https://github.com/mjx-project/mjx/actions/workflows/ci.yml/badge.svg)](https://github.com/mjx-project/mjx/actions/workflows/ci.yml)

# Mjx

**注: このREADMEで書いてある内容はあくまで最終目標で、まだ未実装。前のREADMEはdocsへ移動**

Mjx is a Mahjong environment for AI research and development.
It implements [Japanese (Riichi) Mahjong](https://en.wikipedia.org/wiki/Japanese_Mahjong), one of the most popular variants of Mahjong in the world.

## Features

- **Blazingly fast.** 
  - Mjx is 100x faster than existing OSS simulators. See [benchmark test]().
- **Tenhou compatible.** 
  - The implemented rule is compatible with that of [Tenhou](https://tenhou.net/). See [compatibility tests]().
- **Gym-like API.** 
  - Mjx supports [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) API, which provides [gym](https://github.com/openai/gym)-like API for multi-agent games.
- **Easy distributed computing.**
  - For large-scale reinforcement leraning and evaluation, Mjx supports distributed computing with [gRPC](https://github.com/grpc/grpc).
- **Beautiful visualization.**
  - Mjx supports game log visualization thanks to Tenhou platform.

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


env = mjx.make_env()
env = mjx.TenhouPhonenixRoomReward(env)
env.reset()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = random_select(info.available_actions)
    env.step(action)
```

</td>
<td valign="top">

```c++
#include <mjx/mjx.h>

void main() {
  auto env = mjx::MakeEnv();
  auto env = mjx::TenhouPhoenixRoomReward(env);
  env.Reset();
  for (auto agent: env.AgentIter()) {
    auto [observation, reward, done, info] = env.Last();
    auto action = RandomSelect(info.available_actions);
    env.Step(action);
  }
}
```

</td>
</tr>
</table>

## Available AIs on Mjx
Mjx can translate the protocol between mjx and [mjai](https://github.com/gimite/mjai) using [mjx-mjai-translater](https://github.com/mjx-project/mjx-mjai-translater).
Thus, any AI available with mjai also work with mjx.
Available AIs include:

- MjxBase (A baseline model from Mjx project)
- Bakuuchi
- [gimite/mjai-manue](https://github.com/gimite/mjai-manue) 
- [critter-mj/ankochan](https://github.com/critter-mj/akochan)

## Install

We highly recommend to use docker to use Mjx without environment dependencies.

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
