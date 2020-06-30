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
