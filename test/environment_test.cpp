#include "gtest/gtest.h"
#include <mj/environment.h>
#include <mj/agent_example_rule_based.h>
using namespace mj;

TEST(environment, RunOneRound) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentExampleRuleBased>("agent01"),
            std::make_shared<AgentExampleRuleBased>("agent02"),
            std::make_shared<AgentExampleRuleBased>("agent03"),
            std::make_shared<AgentExampleRuleBased>("agent04")
    };
    Environment env(agents);
    env.RunOneRound();
}

TEST(environment, RunOneGame) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentExampleRuleBased>("agent01"),
            std::make_shared<AgentExampleRuleBased>("agent02"),
            std::make_shared<AgentExampleRuleBased>("agent03"),
            std::make_shared<AgentExampleRuleBased>("agent04")
    };
    Environment env(agents);
    env.RunOneGame();
}
