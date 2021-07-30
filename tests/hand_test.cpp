#include <gtest/gtest.h>
#include <mjx/hand.h>

const std::string sample_json =
    R"({"closedTiles":[0,1,2,3,4,5,6,7,8,9,10,11,12]})";

TEST(hand, Hand) {
  mjxproto::Hand proto;
  google::protobuf::util::JsonStringToMessage(sample_json, &proto);
  auto hand1 = mjx::Hand(proto);
  auto hand2 = mjx::Hand(sample_json);
}

TEST(hand, ToProto) {
  auto hand = mjx::Hand(sample_json);
  const auto& proto = hand.proto();
  std::string json;
  google::protobuf::util::MessageToJsonString(proto, &json);
  EXPECT_EQ(json, sample_json);
}

TEST(hand, ToJson) {
  auto hand = mjx::Hand(sample_json);
  EXPECT_EQ(hand.ToJson(), sample_json);
}

TEST(hand, ShantenNumber) {
  auto hand = mjx::Hand(sample_json);
  EXPECT_EQ(hand.ShantenNumber(), 0);

  hand = mjx::Hand(R"({"closedTiles":[0,1,2,3,4,5,6,7,8,9,10,11,12,12]})");
  EXPECT_EQ(hand.ShantenNumber(), -1);
}

TEST(hand, IsTenpai) {
  auto hand = mjx::Hand(sample_json);
  EXPECT_EQ(hand.IsTenpai(), true);

  // 上がり状態でもテンパイ扱い
  hand = mjx::Hand(R"({"closedTiles":[0,1,2,3,4,5,6,7,8,9,10,11,12,12]})");
  EXPECT_EQ(hand.IsTenpai(), true);
}

TEST(hand, op) {
  auto hand1 = mjx::Hand(sample_json);
  auto hand2 = mjx::Hand(sample_json);
  EXPECT_EQ(hand1, hand2);
  EXPECT_NE(hand1, mjx::Hand());
}
