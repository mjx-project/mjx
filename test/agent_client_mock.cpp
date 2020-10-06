#include "gtest/gtest.h"
#include <mahjong/state.h>
#include <mahjong/agent_client_mock.h>

using namespace mj;

TEST(agent_client_mock, TakeAction) {
    // // random discard
    // auto state = State(9999);
    // std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    // auto drawer = state.UpdateStateByDraw();
    // auto action = agent->TakeAction(state.CreateObservation(drawer));
    // EXPECT_EQ(action.type(), ActionType::kDiscard);
}
