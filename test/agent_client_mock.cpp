#include "gtest/gtest.h"
#include "state.h"
#include "agent_client_mock.h"

using namespace mj;

TEST(agent_client_mock, TakeAction) {
    // random discard
    auto state = State(9999);
    std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    auto drawer = state.UpdateStateByDraw();
    auto observation = state.NewObservation(drawer);
    auto action = agent->TakeAction(std::move(observation));
    EXPECT_EQ(action.type(), ActionType::kDiscard);
}
