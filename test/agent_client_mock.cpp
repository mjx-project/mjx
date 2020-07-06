#include "gtest/gtest.h"
#include "state.h"
#include "agent_client_mock.h"

using namespace mj;

TEST(agent_client_mock, TakeAction) {
    // random discard
    auto state = State(9999);
    state.InitRoundDependentState();
    std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    auto drawer = state.UpdateStateByDraw();
    auto observation = state.mutable_observation(drawer);
    auto action = agent->TakeAction(observation);
    EXPECT_EQ(action.type(), ActionType::kDiscard);
}
