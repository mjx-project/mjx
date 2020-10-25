#include "gtest/gtest.h"
#include <mj/environment.h>
#include <mj/agent_example.h>
using namespace mj;

TEST(environment, RunOneRound) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentExample>("agent01"),
            std::make_shared<AgentExample>("agent02"),
            std::make_shared<AgentExample>("agent03"),
            std::make_shared<AgentExample>("agent04")
    };
    Environment env(agents);
    env.RunOneRound();
}

TEST(environment, RunOneGame) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentExample>("agent01"),
            std::make_shared<AgentExample>("agent02"),
            std::make_shared<AgentExample>("agent03"),
            std::make_shared<AgentExample>("agent04")
    };
    Environment env(agents);
    env.RunOneGame();
}
