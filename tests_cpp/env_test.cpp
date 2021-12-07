#include <mjx/agent.h>
#include <mjx/env.h>
#include <mjx/internal/strategy_rule_based.h>

#include "gtest/gtest.h"

TEST(env, Run) {
  auto agent = std::make_shared<mjx::RandomDebugAgent>();
  std::vector<mjx::PlayerId> player_ids = {"player_0", "player_1", "player_2",
                                           "player_3"};
  std::unordered_map<mjx::PlayerId, mjx::Agent*> agents = {
      {"player_0", agent.get()},
      {"player_1", agent.get()},
      {"player_2", agent.get()},
      {"player_3", agent.get()},
  };

  // Store states
  int num_games = 16;
  int num_parallels = 4;
  auto seed_generator = mjx::RandomSeedGenerator(player_ids);
  mjx::EnvRunner(agents, &seed_generator, num_games, num_parallels);
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
  EXPECT_TRUE(observations.size() == 4);
  auto state = env.state();
  auto player_ids = state.proto().public_observation().player_ids();
  auto tens = state.proto().round_terminal().final_score().tens();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tens[i], expected_tens[player_ids[i]]);
  }
  auto rewards = env.Rewards();
  EXPECT_EQ(rewards["player_0"], 45);
  EXPECT_EQ(rewards["player_1"], 0);
  EXPECT_EQ(rewards["player_2"], -135);
  EXPECT_EQ(rewards["player_3"], 90);
}
