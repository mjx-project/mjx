#include "gtest/gtest.h"
#include <mj/environment.h>
#include <mj/agent_rule_based.h>
using namespace mj;

TEST(environment, RunOneGame) {
    const std::vector<std::shared_ptr<AgentInterface>> agents = {
            std::make_shared<AgentInterfaceLocal>(AgentRuleBased("agent01")),
            std::make_shared<AgentInterfaceLocal>(AgentRuleBased("agent02")),
            std::make_shared<AgentInterfaceLocal>(AgentRuleBased("agent03")),
            std::make_shared<AgentInterfaceLocal>(AgentRuleBased("agent04")),
    };
    Environment env(agents);
    auto result = env.RunOneGame(1234);
    for (const auto& [player_id, ranking]: result.rankings) {
        std::cout << player_id << " " << ranking << " " << result.tens[player_id] << std::endl;
    }

    // Rule based agents have no randomness. Results should be reproducible.
    ASSERT_EQ(result.tens["agent01"], 16800);
    ASSERT_EQ(result.tens["agent02"], 25600);
    ASSERT_EQ(result.tens["agent03"], 26600);
    ASSERT_EQ(result.tens["agent04"], 31000);
}
