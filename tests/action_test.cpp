#include <gtest/gtest.h>
#include <google/protobuf/util/message_differencer.h>
#include <mjx/action.h>
#include <mjx/internal/action.h>

TEST(action, Action) {
  mjxproto::Action proto = mjx::internal::Action::CreateNo(
      mjx::internal::AbsolutePos::kInitEast, "xxx");
  auto action1 = mjx::Action(proto);

  std::string json;
  auto status = google::protobuf::util::MessageToJsonString(proto, &json);
  auto action2 = mjx::Action(json);
}

TEST(action, ToProto) {
  mjxproto::Action proto = mjx::internal::Action::CreateNo(
      mjx::internal::AbsolutePos::kInitEast, "xxx");
  auto action = mjx::Action(proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(action.ToProto(), proto));
}

TEST(action, ToJson) {
  mjxproto::Action proto = mjx::internal::Action::CreateNo(
      mjx::internal::AbsolutePos::kInitEast, "xxx");
  auto action = mjx::Action(proto);
  EXPECT_EQ(action.ToJson(), "{\"gameId\":\"xxx\",\"type\":\"ACTION_TYPE_NO\"}");
}
