#include "gtest/gtest.h"
#include "observation.h"
#include "state.h"

using namespace mj;

TEST(observation, possible_actions) {
    // 第一ツモ後に可能なアクションはdiscardだけで、可能な捨て牌は手牌と一致
    auto state = State(9999);
    state.InitRound();
    auto drawer = state.UpdateStateByDraw();
    auto observation = state.CreateObservation(drawer);
    auto possible_actions = observation.possible_actions();
    EXPECT_EQ(possible_actions.size(), 1);
    EXPECT_EQ(possible_actions.front().type(), ActionType::kDiscard);
    auto hand = state.hand(drawer).ToVector();
    auto possible_discards = possible_actions.front().discard_candidates();
    EXPECT_EQ(hand, possible_discards);

    // TODO(sotetsuk): add more cases
}
