#include "gtest/gtest.h"
#include "state.h"
#include "agent_client_mock.h"

using namespace mj;

TEST(state, InitRound) {
    auto state = State(9999);

    // Hands are different after initializations
    state.InitRoundDependentState();
    auto hand_str1 = state.hands().at(0)->ToString();
    state.InitRoundDependentState();
    auto hand_str2 = state.hands().at(0)->ToString();
    EXPECT_NE(hand_str1, hand_str2);
}

TEST(state, UpdateStateByDraw) {
    auto state = State(9999);
    state.InitRoundDependentState();
    auto drawer = state.UpdateStateByDraw();
    auto hands = state.hands();
    EXPECT_EQ(drawer, AbsolutePos::kEast);
    EXPECT_EQ(hands.at(static_cast<int>(drawer))->Size(), 14);
    EXPECT_EQ(state.Stage(), InRoundStateStage::kAfterDraw);

    // TODO(sotetsuk): add test for different round and turn
}

TEST(state, UpdateStateByAction) {
    // すべてツモとランダムに切るだけでエラーを吐かないか（鳴きなし）
    auto state = State(9999);
    std::unique_ptr<AgentClient> agent = std::make_unique<AgentClientMock>();
    state.InitRoundDependentState();
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(state.mutable_observation(AbsolutePos(j))->possible_actions().size(), 0);
        auto drawer = state.UpdateStateByDraw();
        EXPECT_EQ(drawer, AbsolutePos(i%4));
        auto hand = state.hand(drawer);
        EXPECT_EQ(hand->Size(), 14);
        auto observation = state.mutable_observation(drawer);
        auto action = agent->TakeAction(observation);
        state.UpdateStateByAction(action);
        EXPECT_EQ(hand->Size(), 13);
    }
}
