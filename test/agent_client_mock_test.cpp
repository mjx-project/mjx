#include "gtest/gtest.h"
#include <mjx/state.h>
#include <mjx/agent_example_rule_based.h>
#include <fstream>

using namespace mjx;

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

    State state; Observation observation; mjproto::Action action;
    std::unique_ptr<Agent> agent = std::make_unique<AgentExampleRuleBased>();

    // ツモれるときはツモ
    state = State(GetLastJsonLine("obs-draw-tsumo.json"));
    observation = state.CreateObservations().begin()->second;
    action = agent->TakeAction(std::move(observation));
    EXPECT_EQ(action.type(), mjproto::ACTION_TYPE_TSUMO);

    // リーチできるときはリーチ
    state = State(GetLastJsonLine("obs-draw-riichi.json"));
    observation = state.CreateObservations().begin()->second;
    action = agent->TakeAction(std::move(observation));
    EXPECT_EQ(action.type(), mjproto::ACTION_TYPE_RIICHI);

   // ロンできるときはロン
    state = State(GetLastJsonLine("obs-discard-ron.json"));
    observation = state.CreateObservations().begin()->second;
    action = agent->TakeAction(std::move(observation));
    EXPECT_EQ(action.type(), mjproto::ACTION_TYPE_RON);

    // 九種九牌できるときは流す
    state = State(GetLastJsonLine("obs-draw-kyuusyu.json"));
    observation = state.CreateObservations().begin()->second;
    action = agent->TakeAction(std::move(observation));
    EXPECT_EQ(action.type(), mjproto::ACTION_TYPE_KYUSYU);
}
