#include <mjx/env.h>
#include <mjx/internal/agent_local.h>
#include <mjx/internal/strategy_rule_based.h>
#include <mjx/internal/utils.h>

#include "gtest/gtest.h"
using namespace mjx::internal;

TEST(env, play_one_game) {
  // so large number that the game ends
  int max_cycles = 10000;
  auto env = mjx::Env();
  std::unordered_map<PlayerId, std::string> observations;
  std::unordered_map<PlayerId, int> rewards;
  std::unordered_map<PlayerId, bool> dones;
  std::unordered_map<PlayerId, std::string> infos;

  observations = env.reset();
  StrategyRuleBased strategy;
  for (int i = 0; i < max_cycles; i++) {
    std::unordered_map<PlayerId, std::string> act_dict;
    for (const auto &agent : env.Agents()) {
      if (observations.count(agent) == 0) continue;
      mjxproto::Observation observation;
      // json -> Message
      google::protobuf::util::JsonStringToMessage(observations[agent],
                                                  &observation);
      auto action = strategy.TakeAction(Observation(observation));
      std::string json;
      // Message -> json
      google::protobuf::util::MessageToJsonString(action, &json);
      act_dict[agent] = json;
    }
    std::tie(observations, rewards, dones, infos) =
        env.step(std::move(act_dict));
    if (env.IsGameOver()) break;
  }
  EXPECT_EQ(env.IsGameOver(), true);
}
