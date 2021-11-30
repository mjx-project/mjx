#include <gtest/gtest.h>
#include <mjx/event.h>

const std::string sample_json =
    R"({"type":"EVENT_TYPE_TSUMOGIRI","who":1,"tile":3})";

TEST(event, Event) {
  mjxproto::Event proto;
  google::protobuf::util::JsonStringToMessage(sample_json, &proto);
  auto action1 = mjx::Event(proto);
  auto action2 = mjx::Event(sample_json);
}

TEST(event, ToProto) {
  auto event = mjx::Event(sample_json);
  const auto& proto = event.proto();
  std::string json;
  google::protobuf::util::MessageToJsonString(proto, &json);
  EXPECT_EQ(json, sample_json);
}

TEST(event, ToJson) {
  auto event = mjx::Event(sample_json);
  EXPECT_EQ(event.ToJson(), sample_json);
}

TEST(event, op) {
  auto event1 = mjx::Event(sample_json);
  auto event2 = mjx::Event(sample_json);
  EXPECT_EQ(event1, event2);
  EXPECT_NE(event1, mjx::Event());
}
