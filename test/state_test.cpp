#include "gtest/gtest.h"
#include "state.h"
#include "agent_client_mock.h"

using namespace mj;

TEST(state, InitRound) {
    auto state = State(9999);

    // Hands are different after initializations
    state.InitRound();
    auto hand_str1 = state.hand(AbsolutePos::kInitEast).ToString();
    state.InitRound();
    auto hand_str2 = state.hand(AbsolutePos::kInitEast).ToString();
    EXPECT_NE(hand_str1, hand_str2);
}

TEST(state, UpdateStateByDraw) {
    auto state = State(9999);
    state.InitRound();
    auto drawer = state.UpdateStateByDraw();
    EXPECT_EQ(drawer, AbsolutePos::kInitEast);
    EXPECT_EQ(state.hand(drawer).Size(), 14);

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
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitEast, mj::AbsolutePos::kInitSouth), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitEast, mj::AbsolutePos::kInitWest), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitEast, mj::AbsolutePos::kInitNorth), RelativePos::kLeft);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitSouth, mj::AbsolutePos::kInitEast), RelativePos::kLeft);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitSouth, mj::AbsolutePos::kInitWest), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitSouth, mj::AbsolutePos::kInitNorth), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitWest, mj::AbsolutePos::kInitEast), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitWest, mj::AbsolutePos::kInitSouth), RelativePos::kLeft);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitWest, mj::AbsolutePos::kInitNorth), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitNorth, mj::AbsolutePos::kInitEast), RelativePos::kRight);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitNorth, mj::AbsolutePos::kInitSouth), RelativePos::kMid);
    EXPECT_EQ(State::ToRelativePos(mj::AbsolutePos::kInitNorth, mj::AbsolutePos::kInitWest), RelativePos::kLeft);
}

TEST(state, ToJson) {
    std::string original_json = R"({"playerIds":["-ron-","ASAPIN","うきでん","超ヒモリロ"],"initScore":{"ten":[25000,25000,25000,25000]},"doras":[112],"eventHistory":{"events":[{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":39},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":70},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":125},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":5},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":102},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":114},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":19},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":24},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":90},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":108},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":122},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":17},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":134},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":109},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":116},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":127},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":105},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":100},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":7},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":10},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":26},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":120},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":28},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":98},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":55},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":18},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":15},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_RIICHI","who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_SOUTH","tile":115},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":"ABSOLUTE_POS_INIT_SOUTH"},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":68},{"type":"EVENT_TYPE_CHI","who":"ABSOLUTE_POS_INIT_NORTH","open":42031},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":23},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":34},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":50},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":31},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":20},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":107},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":97},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":30},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":25},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":35},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":60},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":29},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":38},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":59},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":37},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":9},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":27},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":53},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":132},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":67},{},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","tile":110},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":22},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":66},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":69},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":48},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":33},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_WEST","tile":8},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":42},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":96},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":64},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":124},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_NORTH","tile":41},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":21},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":104},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":6},{"who":"ABSOLUTE_POS_INIT_NORTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_NORTH","tile":71},{},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","tile":16},{"who":"ABSOLUTE_POS_INIT_SOUTH"},{"type":"EVENT_TYPE_DISCARD_DRAWN_TILE","who":"ABSOLUTE_POS_INIT_SOUTH","tile":111},{"who":"ABSOLUTE_POS_INIT_WEST"},{"type":"EVENT_TYPE_DISCARD_FROM_HAND","who":"ABSOLUTE_POS_INIT_WEST","tile":4},{"type":"EVENT_TYPE_NO_WINNER"}]},"wall":[48,16,19,34,17,62,79,52,55,30,12,26,120,130,42,67,2,76,13,7,56,57,82,98,31,90,3,4,114,93,5,61,128,1,39,121,32,103,24,70,80,125,66,102,20,108,41,100,87,54,78,84,107,47,14,131,96,51,68,85,28,10,6,18,122,49,134,109,116,127,105,65,92,101,29,23,83,115,77,38,15,43,94,21,50,91,89,45,97,37,25,35,60,132,119,135,59,0,9,27,53,58,118,110,22,124,69,44,33,8,74,129,64,88,72,75,104,73,71,81,111,86,36,99,133,11,40,113,123,95,112,117,46,126,63,106],"uraDoras":[117],"privateInfos":[{"initHand":[48,16,19,34,2,76,13,7,128,1,39,121,87],"draws":[107,96,28,122,116,92,83,15,21,45,35,135,27,110,44,129,75,81]},{"who":"ABSOLUTE_POS_INIT_SOUTH","initHand":[17,62,79,52,56,57,82,98,32,103,24,70,54],"draws":[47,51,10,49,127,101,115,43,50,97,60,59,53,22,33,64,104,111]},{"who":"ABSOLUTE_POS_INIT_WEST","initHand":[55,30,12,26,31,90,3,4,80,125,66,102,78],"draws":[14,68,6,134,105,29,77,94,91,37,132,0,58,124,8,88,73,86]},{"who":"ABSOLUTE_POS_INIT_NORTH","initHand":[120,130,42,67,114,93,5,61,20,108,41,100,84],"draws":[131,85,18,109,65,23,38,89,25,119,9,118,69,74,72,71]}],"terminal":{"noWinner":{"tenpais":[{"who":"ABSOLUTE_POS_INIT_SOUTH","closedTiles":[43,47,49,51,52,54,56,57,62,79,82,101,103]}],"tenChanges":[-1000,3000,-1000,-1000]}}})";
    std::string recovered_json = State(original_json).ToJson();
    EXPECT_EQ(original_json, recovered_json);
    // TODO(sotetsuk): add test cases from Tenhou's log
}