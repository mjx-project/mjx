#include <gtest/gtest.h>
#include <mjx/action.h>

const std::string sample_json = R"({"type":"ACTION_TYPE_NO"})";

TEST(action, Action) {
  mjxproto::Action proto;
  google::protobuf::util::JsonStringToMessage(sample_json, &proto);
  auto action1 = mjx::Action(proto);
  auto action2 = mjx::Action(sample_json);
}

TEST(action, ToProto) {
  auto action = mjx::Action(sample_json);
  const auto& proto = action.proto();
  std::string json;
  google::protobuf::util::MessageToJsonString(proto, &json);
  EXPECT_EQ(json, sample_json);
}

TEST(action, ToJson) {
  auto action = mjx::Action(sample_json);
  EXPECT_EQ(action.ToJson(), sample_json);
}

TEST(action, op) {
  auto action1 = mjx::Action(sample_json);
  auto action2 = mjx::Action(sample_json);
  EXPECT_EQ(action1, action2);
  EXPECT_NE(action1, mjx::Action());
}
