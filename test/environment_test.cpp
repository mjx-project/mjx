#include "gtest/gtest.h"
#include <mj/environment.h>
#include <mj/agent_client_mock.h>
using namespace mj;

TEST(environment, RunOneRound) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentClientMock>("agent01"),
            std::make_shared<AgentClientMock>("agent02"),
            std::make_shared<AgentClientMock>("agent03"),
            std::make_shared<AgentClientMock>("agent04")
    };
    Environment env(agents);
    env.RunOneRound();
}

TEST(environment, RunOneGame) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentClientMock>("agent01"),
            std::make_shared<AgentClientMock>("agent02"),
            std::make_shared<AgentClientMock>("agent03"),
            std::make_shared<AgentClientMock>("agent04")
    };
    Environment env(agents);
    env.RunOneGame();
}
