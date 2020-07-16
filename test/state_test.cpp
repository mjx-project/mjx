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

TEST(state, StealCheck) {
    // TODO(sotetsuk): write here
}

TEST(state, ToRelativePos) {
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kEast, mj::AbsolutePos::kSouth), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kEast, mj::AbsolutePos::kWest), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kEast, mj::AbsolutePos::kNorth), RelativePos::kLeft);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kSouth, mj::AbsolutePos::kEast), RelativePos::kLeft);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kSouth, mj::AbsolutePos::kWest), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kSouth, mj::AbsolutePos::kNorth), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kWest, mj::AbsolutePos::kEast), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kWest, mj::AbsolutePos::kSouth), RelativePos::kLeft);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kWest, mj::AbsolutePos::kNorth), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kNorth, mj::AbsolutePos::kEast), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kNorth, mj::AbsolutePos::kSouth), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kNorth, mj::AbsolutePos::kWest), RelativePos::kLeft);
}