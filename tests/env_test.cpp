#include <mjx/env.h>
#include <mjx/internal/strategy_rule_based.h>

#include "gtest/gtest.h"

TEST(env, RLLibMahjongEnv) {
  auto env = mjx::env::RLlibMahjongEnv();
  std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
      observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards;
  std::unordered_map<mjx::internal::PlayerId, bool> dones;
  std::unordered_map<mjx::internal::PlayerId, std::string> infos;

  env.seed(1234);
  observations = env.reset();
  dones["__all__"] = false;
  auto strategy = mjx::internal::StrategyRuleBased();
  while (!dones.at("__all__")) {
    std::unordered_map<mjx::internal::PlayerId, mjxproto::Action> action_dict;
    for (const auto &[agent, observation] : observations) {
      auto action = strategy.TakeAction(observation);
      action_dict[agent] = action;
    }
    std::tie(observations, rewards, dones, infos) = env.step(action_dict);
  }
  EXPECT_TRUE(dones.at("__all__"));
  EXPECT_EQ(observations["player_0"].round_terminal().final_score().tens()[0], 16800);
  EXPECT_EQ(observations["player_1"].round_terminal().final_score().tens()[1], 25600);
  EXPECT_EQ(observations["player_2"].round_terminal().final_score().tens()[2], 26600);
  EXPECT_EQ(observations["player_3"].round_terminal().final_score().tens()[3], 31000);
}