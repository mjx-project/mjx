#include <gtest/gtest.h>
#include <mjx/state.h>

const std::string sample_json =
    R"({"hiddenState":{"wall":[48,16,19,34,17,62,79,52,55,30,12,26,120,130,42,67,2,76,13,7,56,57,82,98,31,90,3,4,114,93,5,61,128,1,39,121,32,103,24,70,80,125,66,102,20,108,41,100,87,54,78,84,107,47,14,131,96,51,68,85,28,10,6,18,122,49,134,109,116,127,105,65,92,101,29,23,83,115,77,38,15,43,94,21,50,91,89,45,97,37,25,35,60,132,119,135,59,0,9,27,53,58,118,110,22,124,69,44,33,8,74,129,64,88,72,75,104,73,71,81,111,86,36,99,133,11,40,113,123,95,112,117,46,126,63,106],"uraDoraIndicators":[117]},"publicObservation":{"playerIds":["-ron-","ASAPIN","うきでん","超ヒモリロ"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[112],"events":[{"type":"EVENT_TYPE_DRAW"},{"tile":39},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":70},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":125},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":5},{"type":"EVENT_TYPE_DRAW"},{"tile":121},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":32},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":102},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":114},{"type":"EVENT_TYPE_DRAW"},{"tile":19},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":24},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":90},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":108},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":122},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":17},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":134},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":109},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":116},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":127},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":105},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":100},{"type":"EVENT_TYPE_DRAW"},{"tile":7},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":10},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":26},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":120},{"type":"EVENT_TYPE_DRAW"},{"tile":28},{"type":"EVENT_TYPE_DRAW","who":1},{"who":1,"tile":98},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":55},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":18},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":15},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_RIICHI","who":1},{"who":1,"tile":115},{"type":"EVENT_TYPE_RIICHI_SCORE_CHANGE","who":1},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":68},{"type":"EVENT_TYPE_CHI","who":3,"open":42031},{"who":3,"tile":23},{"type":"EVENT_TYPE_DRAW"},{"tile":34},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":50},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":31},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":20},{"type":"EVENT_TYPE_DRAW"},{"tile":107},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":97},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":30},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":25},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":35},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":60},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":29},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":38},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":135},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":59},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":37},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":9},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":27},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":53},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":132},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":67},{"type":"EVENT_TYPE_DRAW"},{"type":"EVENT_TYPE_TSUMOGIRI","tile":110},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":22},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":66},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":69},{"type":"EVENT_TYPE_DRAW"},{"tile":48},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":33},{"type":"EVENT_TYPE_DRAW","who":2},{"type":"EVENT_TYPE_TSUMOGIRI","who":2,"tile":8},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":42},{"type":"EVENT_TYPE_DRAW"},{"tile":96},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":64},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":124},{"type":"EVENT_TYPE_DRAW","who":3},{"who":3,"tile":41},{"type":"EVENT_TYPE_DRAW"},{"tile":21},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":104},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":6},{"type":"EVENT_TYPE_DRAW","who":3},{"type":"EVENT_TYPE_TSUMOGIRI","who":3,"tile":71},{"type":"EVENT_TYPE_DRAW"},{"tile":16},{"type":"EVENT_TYPE_DRAW","who":1},{"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":111},{"type":"EVENT_TYPE_DRAW","who":2},{"who":2,"tile":4},{"type":"EVENT_TYPE_EXHAUSTIVE_DRAW_NORMAL"}]},"privateObservations":[{"initHand":{"closedTiles":[48,16,19,34,2,76,13,7,128,1,39,121,87]},"drawHistory":[107,96,28,122,116,92,83,15,21,45,35,135,27,110,44,129,75,81],"currHand":{"closedTiles":[1,2,13,44,45,75,76,81,83,87,92,128,129]}},{"who":1,"initHand":{"closedTiles":[17,62,79,52,56,57,82,98,32,103,24,70,54]},"drawHistory":[47,51,10,49,127,101,115,43,50,97,60,59,53,22,33,64,104,111],"currHand":{"closedTiles":[43,47,49,51,52,54,56,57,62,79,82,101,103]}},{"who":2,"initHand":{"closedTiles":[55,30,12,26,31,90,3,4,80,125,66,102,78]},"drawHistory":[14,68,6,134,105,29,77,94,91,37,132,0,58,124,8,88,73,86],"currHand":{"closedTiles":[0,3,12,14,58,73,77,78,80,86,88,91,94]}},{"who":3,"initHand":{"closedTiles":[120,130,42,67,114,93,5,61,20,108,41,100,84]},"drawHistory":[131,85,18,109,65,23,38,89,25,119,9,118,69,74,72,71],"currHand":{"closedTiles":[72,74,84,85,89,93,118,119,130,131],"opens":[42031]}}],"roundTerminal":{"finalScore":{"riichi":1,"tens":[24000,27000,24000,24000]},"noWinner":{"tenpais":[{"who":1,"hand":{"closedTiles":[43,47,49,51,52,54,56,57,62,79,82,101,103]}}],"tenChanges":[-1000,3000,-1000,-1000]}}})";

TEST(state, State) {
  mjxproto::State proto;
  google::protobuf::util::JsonStringToMessage(sample_json, &proto);
  auto state1 = mjx::State(proto);
  auto state2 = mjx::State(sample_json);
}

TEST(state, ToProto) {
  auto state = mjx::State(sample_json);
  const auto& proto = state.proto();
  std::string json;
  google::protobuf::util::MessageToJsonString(proto, &json);
  EXPECT_EQ(json, sample_json);
}

TEST(state, ToJson) {
  auto state = mjx::State(sample_json);
  EXPECT_EQ(state.ToJson(), sample_json);
}

TEST(state, op) {
  auto state1 = mjx::State(sample_json);
  auto state2 = mjx::State(sample_json);
  EXPECT_EQ(state1, state2);
  EXPECT_NE(state1, mjx::State());
}

TEST(state, past_decisions) {
  auto state = mjx::State(sample_json);
  auto past_decisions = state.past_decisions();
  EXPECT_EQ(past_decisions.size(),
            87);  // TODO: 87が正しい値かどうかは確認していない
}
