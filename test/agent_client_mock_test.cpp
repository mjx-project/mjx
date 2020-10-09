#include "gtest/gtest.h"
#include <mj/state.h>
#include <mj/agent_client_mock.h>
#include <fstream>

using namespace mj;

TEST(agent_client_mock, TakeAction) {
    // Test utilities
    auto GetLastJsonLine = [](const std::string &filename) {
        auto json_path = std::string(TEST_RESOURCES_DIR) + "/json/" + filename;
        std::ifstream ifs(json_path, std::ios::in);
        std::string buf, json_line;
        while (!ifs.eof()) {
            std::getline(ifs, buf);
            if (buf.empty()) break;
            json_line = buf;
        }
        return json_line;
    };

    auto json = GetLastJsonLine("obs-draw-tsumo.json");
    // // random discard
    // auto state = State(9999);
    // std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    // auto drawer = state.UpdateStateByDraw();
    // auto action = agent->TakeAction(state.CreateObservation(drawer));
    // EXPECT_EQ(action.type(), ActionType::kDiscard);
}
