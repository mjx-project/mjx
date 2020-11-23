#include "gtest/gtest.h"
#include <mj/environment.h>
#include <mj/agent_example_rule_based.h>
using namespace mj;

TEST(environment, RunOneGame) {
    const std::vector<std::shared_ptr<Agent>> agents = {
            std::make_shared<AgentExampleRuleBased>("agent01"),
            std::make_shared<AgentExampleRuleBased>("agent02"),
            std::make_shared<AgentExampleRuleBased>("agent03"),
            std::make_shared<AgentExampleRuleBased>("agent04")
    };
    Environment env(agents);
    auto result = env.RunOneGame(1234);
    for (const auto& [player_id, ranking]: result.rankings) {
        std::cout << player_id << " " << ranking << " " << result.tens[player_id] << std::endl;
    }
}
