#include "gtest/gtest.h"
#include <mj/environment.h>
#include <mj/rule_based_agent.h>
using namespace mj;

TEST(environment, RunOneRound) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<RuleBasedAgent>("agent01"),
            std::make_shared<RuleBasedAgent>("agent02"),
            std::make_shared<RuleBasedAgent>("agent03"),
            std::make_shared<RuleBasedAgent>("agent04")
    };
    Environment env(agents);
    env.RunOneRound();
}

TEST(environment, RunOneGame) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<RuleBasedAgent>("agent01"),
            std::make_shared<RuleBasedAgent>("agent02"),
            std::make_shared<RuleBasedAgent>("agent03"),
            std::make_shared<RuleBasedAgent>("agent04")
    };
    Environment env(agents);
    env.RunOneGame();
}
