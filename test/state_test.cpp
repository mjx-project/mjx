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
    std::string original_json = R"({"playerIds":["-ron-","ASAPIN","うきでん","超ヒモリロ"],"initScore":{"ten":[25000,25000,25000,25000]},"doras":[112],"initHands":[{"tiles":[48,16,19,34,2,76,13,7,128,1,39,121,87]},{"who":"ABSOLUTE_POS_INIT_SOUTH","tiles":[17,62,79,52,56,57,82,98,32,103,24,70,54]},{"who":"ABSOLUTE_POS_INIT_WEST","tiles":[55,30,12,26,31,90,3,4,80,125,66,102,78]},{"who":"ABSOLUTE_POS_INIT_NORTH","tiles":[120,130,42,67,114,93,5,61,20,108,41,100,84]}],"eventHistory":{"events":[{"tile":107},{"type":"EVENT_TYPE_DISCARD","tile":39},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":47},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":70},{"who":"ABSOLUTE_POS_INIT_WEST","tile":14},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":125},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":131},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":5},{"tile":96},{"type":"EVENT_TYPE_DISCARD","tile":121},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":51},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":32},{"who":"ABSOLUTE_POS_INIT_WEST","tile":68},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":102},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":85},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":114},{"tile":28},{"type":"EVENT_TYPE_DISCARD","tile":19},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":10},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":24},{"who":"ABSOLUTE_POS_INIT_WEST","tile":6},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":90},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":18},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":108},{"tile":122},{"type":"EVENT_TYPE_DISCARD","tile":122},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":49},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":17},{"who":"ABSOLUTE_POS_INIT_WEST","tile":134},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":134},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":109},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":109},{"tile":116},{"type":"EVENT_TYPE_DISCARD","tile":116},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":127},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":127},{"who":"ABSOLUTE_POS_INIT_WEST","tile":105},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":105},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":65},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":100},{"tile":92},{"type":"EVENT_TYPE_DISCARD","tile":7},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":101},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":10},{"who":"ABSOLUTE_POS_INIT_WEST","tile":29},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":26},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":23},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":120},{"tile":83},{"type":"EVENT_TYPE_DISCARD","tile":28},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":115},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":98},{"who":"ABSOLUTE_POS_INIT_WEST","tile":77},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":55},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":38},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":18},{"tile":15},{"type":"EVENT_TYPE_DISCARD","tile":15},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":43},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_RIICHI"},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":115},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_RIICHI_SCORE_CHANGE"},{"who":"ABSOLUTE_POS_INIT_WEST","tile":94},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":68},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_CHI","open":42031},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":23},{"tile":21},{"type":"EVENT_TYPE_DISCARD","tile":34},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":50},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":50},{"who":"ABSOLUTE_POS_INIT_WEST","tile":91},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":31},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":89},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":20},{"tile":45},{"type":"EVENT_TYPE_DISCARD","tile":107},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":97},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":97},{"who":"ABSOLUTE_POS_INIT_WEST","tile":37},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":30},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":25},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":25},{"tile":35},{"type":"EVENT_TYPE_DISCARD","tile":35},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":60},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":60},{"who":"ABSOLUTE_POS_INIT_WEST","tile":132},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":29},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":119},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":38},{"tile":135},{"type":"EVENT_TYPE_DISCARD","tile":135},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":59},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":59},{"who":"ABSOLUTE_POS_INIT_WEST"},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":37},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":9},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":9},{"tile":27},{"type":"EVENT_TYPE_DISCARD","tile":27},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":53},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":53},{"who":"ABSOLUTE_POS_INIT_WEST","tile":58},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":132},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":118},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":67},{"tile":110},{"type":"EVENT_TYPE_DISCARD","tile":110},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":22},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":22},{"who":"ABSOLUTE_POS_INIT_WEST","tile":124},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":66},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":69},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":69},{"tile":44},{"type":"EVENT_TYPE_DISCARD","tile":48},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":33},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":33},{"who":"ABSOLUTE_POS_INIT_WEST","tile":8},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":8},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":74},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":42},{"tile":129},{"type":"EVENT_TYPE_DISCARD","tile":96},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":64},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":64},{"who":"ABSOLUTE_POS_INIT_WEST","tile":88},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":124},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":72},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":41},{"tile":75},{"type":"EVENT_TYPE_DISCARD","tile":21},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":104},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":104},{"who":"ABSOLUTE_POS_INIT_WEST","tile":73},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":6},{"who":"ABSOLUTE_POS_INIT_NORTH","tile":71},{"who":"ABSOLUTE_POS_INIT_NORTH","type":"EVENT_TYPE_DISCARD","tile":71},{"tile":81},{"type":"EVENT_TYPE_DISCARD","tile":16},{"who":"ABSOLUTE_POS_INIT_SOUTH","tile":111},{"who":"ABSOLUTE_POS_INIT_SOUTH","type":"EVENT_TYPE_DISCARD","tile":111},{"who":"ABSOLUTE_POS_INIT_WEST","tile":86},{"who":"ABSOLUTE_POS_INIT_WEST","type":"EVENT_TYPE_DISCARD","tile":4}]},"wall":[48,16,19,34,17,62,79,52,55,30,12,26,120,130,42,67,2,76,13,7,56,57,82,98,31,90,3,4,114,93,5,61,128,1,39,121,32,103,24,70,80,125,66,102,20,108,41,100,87,54,78,84,107,47,14,131,96,51,68,85,28,10,6,18,122,49,134,109,116,127,105,65,92,101,29,23,83,115,77,38,15,43,94,21,50,91,89,45,97,37,25,35,60,132,119,135,59,0,9,27,53,58,118,110,22,124,69,44,33,8,74,129,64,88,72,75,104,73,71,81,111,86,36,99,133,11,40,113,123,95,112,117,46,126,63,106],"uraDoras":[117],"endInfo":{"noWinnerEnd":{"tenpais":[{"who":"ABSOLUTE_POS_INIT_SOUTH","closedTiles":[43,47,49,51,52,54,56,57,62,79,82,101,103]}],"tenChanges":[-1000,3000,-1000,-1000]}}})";
    std::string recovered_json = State(original_json).ToJson();
    EXPECT_EQ(original_json, recovered_json);
    // TODO(sotetsuk): add test cases from Tenhou's log
}