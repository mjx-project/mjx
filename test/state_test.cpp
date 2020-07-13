#include "gtest/gtest.h"
#include "state.h"
#include "agent_client_mock.h"

using namespace mj;

TEST(state, InitRound) {
    auto state = State(9999);

    // Hands are different after initializations
    state.InitRound();
    auto hand_str1 = state.hand(AbsolutePos::kEast).ToString();
    state.InitRound();
    auto hand_str2 = state.hand(AbsolutePos::kEast).ToString();
    EXPECT_NE(hand_str1, hand_str2);
}

TEST(state, UpdateStateByDraw) {
    auto state = State(9999);
    state.InitRound();
    auto drawer = state.UpdateStateByDraw();
    EXPECT_EQ(drawer, AbsolutePos::kEast);
    EXPECT_EQ(state.hand(drawer).Size(), 14);
    EXPECT_EQ(state.stage(), RoundStage::kAfterDraw);

    // TODO(sotetsuk): add test for different round and turn
}

TEST(state, UpdateStateByAction) {
    // すべてツモとランダムに切るだけでエラーを吐かないか（鳴きなし）
    auto state = State(9999);
    std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    state.InitRound();
    for (int i = 0; i < 50; ++i) {
        auto drawer = state.UpdateStateByDraw();
        EXPECT_EQ(drawer, AbsolutePos(i%4));
        EXPECT_EQ(state.hand(drawer).Size(), 14);
        auto action = agent->TakeAction(state.CreateObservation(drawer));
        state.UpdateStateByAction(action);
        EXPECT_EQ(state.hand(drawer).Size(), 13);
    }
}

TEST(state, RonCheck) {
    // TODO(sotetsuk): write here
}