#include "gtest/gtest.h"
#include "yaku_evaluator.h"

#include "types.h"
#include "hand.h"

using namespace mj;

class YakuTest : public ::testing::Test {
protected:
    // virtual void SetUp() {}
    // virtual void TearDown() {}

    YakuEvaluator evaluator;
};


TEST_F(YakuTest, FullyConcealdHand)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku1.count(Yaku::kFullyConcealedHand), 1);

    // ロンはダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Ron("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kFullyConcealedHand), 0);

    // 鳴きはダメ
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9").Pon("p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.count(Yaku::kFullyConcealedHand), 0);
}

TEST_F(YakuTest, Pinfu)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku1.count(Yaku::kPinfu), 1);

    // 刻子はダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kPinfu), 0);

    // 鳴きはダメ
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9").Pon("p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.count(Yaku::kPinfu), 0);

    // 役牌の雀頭はダメ
    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,s7,s8,s9,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku4.count(Yaku::kPinfu), 0);

    // リャンメン待ち以外はダメ
    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m6,m9,m9,s7,s8,s9,p1,p2,p3").Tsumo("m5")));
    EXPECT_EQ(yaku5.count(Yaku::kPinfu), 0);
    auto yaku6 = evaluator.Eval(
            Hand(HandParams("m1,m2,m4,m5,m6,m9,m9,s7,s8,s9,p1,p2,p3").Tsumo("m3")));
    EXPECT_EQ(yaku6.count(Yaku::kPinfu), 0);
    auto yaku7 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,m9,s7,s8,s9,p1,p2,p3").Tsumo("m9")));
    EXPECT_EQ(yaku7.count(Yaku::kPinfu), 0);
}
