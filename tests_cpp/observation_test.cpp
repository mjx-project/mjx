#include <gtest/gtest.h>
#include <mjx/observation.h>

const std::string sample_json =
    R"({"who":1,"publicObservation":{"playerIds":["rule-based-0","target-player","rule-based-2","rule-based-3"],"initScore":{"tens":[25000,25000,25000,25000]},"doraIndicators":[114]},"privateObservation":{"who":1,"initHand":{"closedTiles":[123,22,86,7,98,64,53,36,109,61,122,116,117]},"currHand":{"closedTiles":[7,22,36,53,61,64,86,98,109,116,117,122,123]}},"legalActions":[{"type":"ACTION_TYPE_DUMMY","who":1}]})";

TEST(observation, Observation) {
  mjxproto::Observation proto;
  google::protobuf::util::JsonStringToMessage(sample_json, &proto);
  auto observation1 = mjx::Observation(proto);
  auto observation2 = mjx::Observation(sample_json);
}

TEST(observation, ToProto) {
  auto observation = mjx::Observation(sample_json);
  const auto& proto = observation.proto();
  std::string json;
  google::protobuf::util::MessageToJsonString(proto, &json);
  EXPECT_EQ(json, sample_json);
}

TEST(observation, ToJson) {
  auto observation = mjx::Observation(sample_json);
  EXPECT_EQ(observation.ToJson(), sample_json);
}

TEST(observation, op) {
  auto observation1 = mjx::Observation(sample_json);
  auto observation2 = mjx::Observation(sample_json);
  EXPECT_EQ(observation1, observation2);
  EXPECT_NE(observation1, mjx::Observation());
}
