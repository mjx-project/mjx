import random

import _mjx

import mjx.agents
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
    random_agent = mjx.agents.RandomDebugAgent()

    random.seed(1234)
    env = mjx.env.MjxEnv()
    obs_dict = env.reset(1234)

    assert len(obs_dict) == 1
    assert "player_2" in obs_dict
    assert env.state().to_proto().hidden_state.wall[:5] == [24, 3, 87, 124, 97]

    while not env.done():
        action_dict = {}
        for agent, obs in obs_dict.items():
            action_dict[agent] = random_agent.act(obs)
        obs_dict = env.step(action_dict)
        if not env.done():
            rewards = env.rewards()
            assert len(rewards) == 4
            assert rewards["player_0"] == 0
            assert rewards["player_1"] == 0
            assert rewards["player_2"] == 0
            assert rewards["player_3"] == 0

    # MjxEnvのDone周りの挙動についての仕様は
    # https://github.com/mjx-project/mjx/pull/1055
    # を参考
    # 基本的な考えとしては
    #  - Dummyアクションは情報共有のためにあるので、Dummyを含む
    #    Obsが返ってくるタイミングで、doneがtrue, non-zero rewardsが得られる
    #  - Dummyアクションを取ると、次の局に進む（オーラス）は空obsが返ってくる
    #  - ユーザ側から見れば、doneがtrueのタイミングでobsやrewardsにアクセスすれば、
    #    終局時の情報が得られる
    assert env.done("game")
    assert env.done("round")
    assert len(obs_dict) == 4
    for _, obs in obs_dict.items():
        assert len(obs.legal_actions()) == 1
        assert obs.legal_actions()[0].type() == mjx.ActionType.DUMMY
    rewards = env.rewards()

    assert len(rewards) == 4
    assert rewards["player_0"] == 90
    assert rewards["player_1"] == 0
    assert rewards["player_2"] == 45
    assert rewards["player_3"] == -135

    # if dummy actions are taken after done("game")
    # 基本的にはdone("game")がTrueになったらresetを呼ぶ想定
    # done("game")がtrueになってdummyをstepで呼んだ後の値は使われる想定ではないが
    # 今の定義は、他のdone("hand")がtrueになってdummyをstepに渡した直後との整合性を取るため、
    # rewardは全て0で、doneもすべてFalse
    # ただ、done()やrewards()が呼ばれたらexceptionを投げるように変更するかもしれない
    action_dict = {}
    for agent, obs in obs_dict.items():
        action_dict[agent] = random_agent.act(obs)
    assert len(action_dict) == 4
    obs_dict = env.step(action_dict)
    assert obs_dict == {}
    # TODO: raise exceptions or print warning to ask user to call reset
    rewards = env.rewards()
    assert len(rewards) == 4
    assert rewards["player_0"] == 0
    assert rewards["player_1"] == 0
    assert rewards["player_2"] == 0
    assert rewards["player_3"] == 0
    assert not env.done()  # done == Trueとなるタイミングは各局一度だけ
    assert not env.done("round")  # done == Trueとなるタイミングは各局一度だけ

    # test specifying dealer order
    obs_dict = env.reset(1234, ["player_3", "player_1", "player_2", "player_0"])
    assert len(obs_dict) == 1
    assert "player_3" in obs_dict

    obs_dict = env.reset(1234)
    assert len(obs_dict) == 1
    assert "player_2" in obs_dict


def testMjxEnvRewardsHandWin():
    random_agent = mjx.agents.RuleBasedAgent()
    random.seed(1234)
    env = mjx.env.MjxEnv()
    obs_dict = env.reset(1234)
    while not env.done("round"):
        action_dict = {}
        for agent, obs in obs_dict.items():
            action_dict[agent] = random_agent.act(obs)
        obs_dict = env.step(action_dict)
        if not env.done("round"):
            rewards = env.rewards("round_win")
            assert len(rewards) == 4
            assert rewards["player_0"] == 0
            assert rewards["player_1"] == 0
            assert rewards["player_2"] == 0
            assert rewards["player_3"] == 0

    rewards = env.rewards("round_win")
    assert len(rewards) == 4
    assert rewards["player_0"] == 0, env.state().to_json()
    assert rewards["player_1"] == 0, env.state().to_json()
    assert rewards["player_2"] == 0, env.state().to_json()
    assert rewards["player_3"] == 1, env.state().to_json()


def testMjxEnvRoundDone():
    random_agent = mjx.agents.RandomDebugAgent()
    random.seed(1234)
    env = mjx.env.MjxEnv()
    obs_dict = env.reset(1234)
    while not env.done(done_type="round"):
        action_dict = {}
        for agent, obs in obs_dict.items():
            action_dict[agent] = random_agent.act(obs)
        obs_dict = env.step(action_dict)
    assert not env.done("game")
    assert env.done("round")
    assert len(obs_dict) == 4
    for _, obs in obs_dict.items():
        assert len(obs.legal_actions()) == 1
        assert obs.legal_actions()[0].type() == mjx.ActionType.DUMMY
