#include <mjx/internal/agent.h>
#include <mjx/internal/mjx.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(state, ForbidFifthKan) {
  // https://github.com/mjx-project/mjx/pull/701
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
  env.RunOneGame(13762514072779568829);
}