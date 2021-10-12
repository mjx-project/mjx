import random

import _mjx

import mjx.agent
import mjx.env


def test_EnvRunner():
    agents = {
        "player_0": _mjx.RandomDebugAgent(),
        "player_1": _mjx.RandomDebugAgent(),
        "player_2": _mjx.RandomDebugAgent(),
        "player_3": _mjx.RandomDebugAgent(),
    }
    seed_generator = _mjx.RandomSeedGenerator(["player_0", "player_1", "player_2", "player_3"])
    runner = _mjx.EnvRunner(agents, seed_generator, 100, 4, 10, None, None)


def test_RandomSeedGenerator():
    seed_generator = _mjx.RandomSeedGenerator(["player_0", "player_1", "player_2", "player_3"])  # type: ignore
    N = 100000
    seeds = set([])
    first_dealer_cnt = {}
    for i in range(N):
        seed, player_ids = seed_generator.get()
        seeds.add(seed)
        first_dealer_cnt[player_ids[0]] = (
            first_dealer_cnt[player_ids[0]] + 1 if player_ids[0] in first_dealer_cnt else 1
        )

    assert len(seeds) == N
    assert N / 4 - N / 10 < first_dealer_cnt["player_0"] < N / 4 + N / 10
    assert N / 4 - N / 10 < first_dealer_cnt["player_1"] < N / 4 + N / 10
    assert N / 4 - N / 10 < first_dealer_cnt["player_2"] < N / 4 + N / 10
    assert N / 4 - N / 10 < first_dealer_cnt["player_3"] < N / 4 + N / 10


def test_DuplicateRandomSeedGenerator():
    seed_generator = _mjx.DuplicateRandomSeedGenerator(["player_0", "player_1", "player_2", "player_3"])  # type: ignore
    N = 100000
    seeds = set([])
    first_dealer_cnt = {}
    for i in range(N):
        seed, player_ids = seed_generator.get()
        seeds.add(seed)
        first_dealer_cnt[player_ids[0]] = (
            first_dealer_cnt[player_ids[0]] + 1 if player_ids[0] in first_dealer_cnt else 1
        )

    assert len(seeds) == N // 4
    assert N // 4 == first_dealer_cnt["player_0"]
    assert N // 4 == first_dealer_cnt["player_1"]
    assert N // 4 == first_dealer_cnt["player_2"]
    assert N // 4 == first_dealer_cnt["player_3"]


def test_MjxEnv():
    random_agent = mjx.agent.RandomDebugAgent()

    random.seed(1234)
    env = mjx.env.MjxEnv()
    obs_dict = env.reset(1234)

    assert len(obs_dict) == 1
    assert "player_2" in obs_dict
    assert env.state.to_proto().hidden_state.wall[:5] == [24, 3, 87, 124, 97]

    while not env.done():
        action_dict = {}
        for agent, obs in obs_dict.items():
            action_dict[agent] = random_agent.act(obs)
        obs_dict = env.step(action_dict)
    rewards = env.rewards()

    assert len(rewards) == 4
    assert rewards["player_0"] == 90
    assert rewards["player_1"] == 0
    assert rewards["player_2"] == 45
    assert rewards["player_3"] == -135

    # test specifing daler order
    obs_dict = env.reset(1234, ["player_3", "player_1", "player_2", "player_0"])
    assert len(obs_dict) == 1
    assert "player_3" in obs_dict

    obs_dict = env.reset(1234)
    assert len(obs_dict) == 1
    assert "player_2" in obs_dict
