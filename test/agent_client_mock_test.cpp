#include "gtest/gtest.h"
#include <mj/state.h>
#include <mj/agent_client_mock.h>

using namespace mj;

// Test utilities
std::string GetLastJsonLine(const std::string &filename) {
    auto json_path = std::string(TEST_RESOURCES_DIR) + "/json/" + filename;
    std::ifstream ifs(json_path, std::ios::in);
    std::string buf, json_line;
    while (!ifs.eof()) {
        std::getline(ifs, buf);
        if (buf.empty()) break;
        json_line = buf;
    }
    return json_line;
}

TEST(agent_client_mock, TakeAction) {
    State state; Action action;
    std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    // auto action = agent->TakeAction(state.CreateObservation(drawer));
    // EXPECT_EQ(action.type(), ActionType::kDiscard);
}
