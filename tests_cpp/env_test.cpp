#include <mjx/agent.h>
#include <mjx/env.h>
#include <mjx/internal/strategy_rule_based.h>

#include "gtest/gtest.h"

TEST(env, run) {
  auto agent = std::make_shared<mjx::RandomDebugAgent>();
  std::unordered_map<mjx::PlayerId, mjx::Agent*> agents = {
      {"player_0", agent.get()},
      {"player_1", agent.get()},
      {"player_2", agent.get()},
      {"player_3", agent.get()},
  };
  auto runner = mjx::EnvRunner(agents);
  runner.Run();
}

TEST(env, MjxEnv) {
  std::unordered_map<mjx::PlayerId, mjx::Observation> observations;
  auto strategy = mjx::internal::StrategyRuleBased();
  std::unordered_map<mjx::internal::PlayerId, int> expected_tens = {
      {"player_0", 26600},
      {"player_1", 25600},
      {"player_2", 16800},
      {"player_3", 31000}};

  auto env = mjx::MjxEnv();
  observations = env.Reset(1234);
  while (!env.Done()) {
    {
      std::unordered_map<mjx::PlayerId, mjx::Action> action_dict;
      for (const auto& [agent, observation] : observations) {
        auto action = strategy.TakeAction(observation.proto());
        action_dict[agent] = mjx::Action(action);
      }
      observations = env.Step(action_dict);
    }
  }
  auto player_ids =
      observations["player_0"].proto().public_observation().player_ids();
  auto tens =
      observations["player_0"].proto().round_terminal().final_score().tens();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tens[i], expected_tens[player_ids[i]]);
  }
  auto rewards = env.Rewards();
  EXPECT_EQ(rewards["player_0"], 45);
  EXPECT_EQ(rewards["player_1"], 0);
  EXPECT_EQ(rewards["player_2"], -135);
  EXPECT_EQ(rewards["player_3"], 90);
}

TEST(env, RLlibMahjongEnv) {
  auto env = mjx::RLlibMahjongEnv();
  std::unordered_map<mjx::internal::PlayerId, mjx::Observation> observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards;
  std::unordered_map<mjx::internal::PlayerId, bool> dones;
  std::unordered_map<mjx::internal::PlayerId, std::string> infos;

  env.Seed(1234);
  observations = env.Reset();
  dones["__all__"] = false;
  auto strategy = mjx::internal::StrategyRuleBased();
  while (!dones.at("__all__")) {
    std::unordered_map<mjx::internal::PlayerId, mjx::Action> action_dict;
    for (const auto& [agent, observation] : observations) {
      auto is_round_over = observation.proto().has_round_terminal();
      auto action = strategy.TakeAction(observation.proto());
      action_dict[agent] = mjx::Action(action);
      if (!is_round_over) EXPECT_NE(action.type(), mjxproto::ACTION_TYPE_DUMMY);
      if (is_round_over) EXPECT_EQ(action.type(), mjxproto::ACTION_TYPE_DUMMY);
    }
    std::tie(observations, rewards, dones, infos) = env.Step(action_dict);
  }
  EXPECT_TRUE(dones.at("__all__"));
  auto player_ids =
      observations["player_0"].proto().public_observation().player_ids();
  std::unordered_map<mjx::internal::PlayerId, int> expected_tens = {
      {"player_0", 26600},
      {"player_1", 25600},
      {"player_2", 16800},
      {"player_3", 31000}};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(
        observations["player_0"].proto().round_terminal().final_score().tens(i),
        expected_tens[player_ids[i]]);
  }
  for (const auto& [player_id, obs] : observations) {
    EXPECT_EQ(obs.proto().legal_actions().size(), 1);
    EXPECT_EQ(obs.proto().legal_actions(0).type(), mjxproto::ACTION_TYPE_DUMMY);
  }
  EXPECT_EQ(rewards.at("player_0"), 45);
  EXPECT_EQ(rewards.at("player_1"), 0);
  EXPECT_EQ(rewards.at("player_2"), -135);
  EXPECT_EQ(rewards.at("player_3"), 90);
}

TEST(env, PettingZooMahjongEnv) {
  auto strategy = mjx::internal::StrategyRuleBased();
  std::optional<mjx::PlayerId> agent_selection;
  std::unordered_map<mjx::internal::PlayerId, int> expected_tens = {
      {"player_0", 26600},
      {"player_1", 25600},
      {"player_2", 16800},
      {"player_3", 31000}};

  auto env = mjx::PettingZooMahjongEnv();
  env.Seed(1234);
  env.Reset();
  agent_selection = env.agent_selection();
  while (agent_selection) {
    auto [observation, reward, done, info] = env.Last();
    auto action = strategy.TakeAction(observation.value().proto());
    env.Step(mjx::Action(action));

    // std::cerr << agent_selection.value() << ", " << reward << ", " << done <<
    // ", " << observation.value().ToJson() << std::endl;
    if (done) {
      if (agent_selection.value() == "player_0") EXPECT_EQ(reward, 45);
      if (agent_selection.value() == "player_1") EXPECT_EQ(reward, 0);
      if (agent_selection.value() == "player_2") EXPECT_EQ(reward, -135);
      if (agent_selection.value() == "player_3") EXPECT_EQ(reward, 90);
      auto player_ids =
          observation.value().proto().public_observation().player_ids();
      auto tens =
          observation.value().proto().round_terminal().final_score().tens();
      for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(tens[i], expected_tens[player_ids[i]]);
      }
    }

    agent_selection = env.agent_selection();
  }
}
