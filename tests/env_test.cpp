#include <mjx/env.h>
#include <mjx/internal/strategy_rule_based.h>

#include "gtest/gtest.h"

TEST(env, RLLibMahjongEnv) {
  int max_cycles = 10000;

  auto env = mjx::env::RLlibMahjongEnv();
  std::unordered_map<mjx::internal::PlayerId, mjxproto::Observation>
      observations;
  std::unordered_map<mjx::internal::PlayerId, int> rewards;
  std::unordered_map<mjx::internal::PlayerId, bool> dones;
  std::unordered_map<mjx::internal::PlayerId, std::string> infos;

  observations = env.reset();
  dones["__all__"] = false;
  auto strategy = mjx::internal::StrategyRuleBased();
  for (int i = 0; i < max_cycles; i++) {
    std::unordered_map<mjx::internal::PlayerId, mjxproto::Action> action_dict;
    for (const auto &[agent, observation] : observations) {
      auto action = strategy.TakeAction(observation);
      action_dict[agent] = action;
    }
    std::tie(observations, rewards, dones, infos) = env.step(action_dict);
    if (dones.at("__all__")) break;
  }
  EXPECT_TRUE(dones.at("__all__"));
}
