#include <mjx/agent_local.h>
#include <mjx/environment.h>
#include <mjx/strategy_rule_based.h>

#include "gtest/gtest.h"
using namespace mjx;

TEST(environment, RunOneGame) {
  const std::vector<std::shared_ptr<Agent>> agents = {
      std::make_shared<AgentLocal>("agent01",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent02",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent03",
                                   std::make_unique<StrategyRuleBased>()),
      std::make_shared<AgentLocal>("agent04",
                                   std::make_unique<StrategyRuleBased>())};
  Environment env(agents);
  auto result = env.RunOneGame(1234);
  for (const auto& [player_id, ranking] : result.rankings) {
    std::cout << player_id << " " << ranking << " " << result.tens[player_id]
              << std::endl;
  }

  // Rule based agents have no randomness. Results should be reproducible.
  ASSERT_EQ(result.tens["agent01"], 16800);
  ASSERT_EQ(result.tens["agent02"], 25600);
  ASSERT_EQ(result.tens["agent03"], 26600);
  ASSERT_EQ(result.tens["agent04"], 31000);
}
