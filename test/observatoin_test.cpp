#include "gtest/gtest.h"
#include "observation.h"
#include "state.h"

using namespace mj;

TEST(observation, possible_actions) {
    // 第一ツモ後に可能なアクションはdiscardだけ
    auto state = State(9999);
    auto drawer = state.UpdateStateByDraw();
    auto observation = state.CreateObservation(drawer);
    auto possible_actions = observation.possible_actions();
    EXPECT_EQ(possible_actions.size(), 1);
    EXPECT_EQ(possible_actions.front().type(), ActionType::kDiscard);

    // TODO(sotetsuk): add more cases
}
