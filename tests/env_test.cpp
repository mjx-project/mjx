#include <mjx/env.h>
#include <mjx/internal/strategy_rule_based.h>

#include "gtest/gtest.h"

TEST(env, RLlibMahjongEnv) {
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
    for (const auto& [agent, observation] : observations) {
      auto action = strategy.TakeAction(observation);
      action_dict[agent] = action;
      EXPECT_NE(action.type(), mjxproto::ACTION_TYPE_DUMMY);
    }
    std::tie(observations, rewards, dones, infos) = env.step(action_dict);
  }
  EXPECT_TRUE(dones.at("__all__"));
  auto player_ids = observations["player_0"].public_observation().player_ids();
  std::unordered_map<mjx::internal::PlayerId, int> expected_tens = {
      {"player_0", 26600},
      {"player_1", 25600},
      {"player_2", 16800},
      {"player_3", 31000}};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(observations["player_0"].round_terminal().final_score().tens(i),
              expected_tens[player_ids[i]]);
  }
  for (const auto& [player_id, obs] : observations) {
    EXPECT_EQ(obs.possible_actions().size(), 1);
    EXPECT_EQ(obs.possible_actions(0).type(), mjxproto::ACTION_TYPE_DUMMY);
  }
}
