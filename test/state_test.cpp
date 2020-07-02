#include "gtest/gtest.h"
#include "state.h"

using namespace mj;

TEST(state, InitRound) {
    auto state = State(9999);

    // Hands are different after initializations
    state.InitRound();
    auto hand_str1 = state.GetHands().at(0).ToString();
    state.InitRound();
    auto hand_str2 = state.GetHands().at(0).ToString();
    EXPECT_NE(hand_str1, hand_str2);
}

TEST(state, UpdateStateByDraw) {
    auto state = State(9999);
    state.InitRound();
    auto drawer = state.UpdateStateByDraw();
    const auto &hands = state.GetHands();
    EXPECT_EQ(drawer, AbsolutePos::kEast);
    EXPECT_EQ(hands.at(static_cast<int>(drawer)).Size(), 14);
    EXPECT_EQ(state.Stage(), InRoundStateStage::kAfterDraw);

    // TODO(sotetsuk): add test for different round and turn
}
