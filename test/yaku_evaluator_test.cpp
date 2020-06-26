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
    EXPECT_EQ(yaku1[Yaku::kFullyConcealedHand], 1);

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
    EXPECT_EQ(yaku1[Yaku::kPinfu], 1);

    // 鳴きはダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9").Pon("p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kPinfu), 0);

    // 刻子はダメ
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.count(Yaku::kPinfu), 0);

    // 役牌の雀頭はダメ
    // TODO: 場風, 自風も弾く
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

TEST_F(YakuTest, AllSimples) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,p2,p3,p4").Tsumo("m6")));
    EXPECT_EQ(yaku1.count(Yaku::kAllSimples), 1);
    EXPECT_EQ(yaku1[Yaku::kAllSimples], 1);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kAllSimples), 0);
}

TEST_F(YakuTest, Dragon) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,wd,wd,wd").Tsumo("m6")));
    EXPECT_EQ(yaku1.count(Yaku::kWhiteDragon), 1);
    EXPECT_EQ(yaku1[Yaku::kWhiteDragon], 1);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,gd,gd,gd").Tsumo("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kGreenDragon), 1);
    EXPECT_EQ(yaku2[Yaku::kGreenDragon], 1);

    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,rd,rd,rd").Tsumo("m6")));
    EXPECT_EQ(yaku3.count(Yaku::kRedDragon), 1);
    EXPECT_EQ(yaku3[Yaku::kRedDragon], 1);

    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku4.count(Yaku::kWhiteDragon), 0);
    EXPECT_EQ(yaku4.count(Yaku::kGreenDragon), 0);
    EXPECT_EQ(yaku4.count(Yaku::kRedDragon), 0);
}

TEST_F(YakuTest, AllTermsAndHonours)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m9,m9,s1,s1,ew,ew,ew,rd,rd,rd").Tsumo("m9")));
    EXPECT_EQ(yaku1.count(Yaku::kAllTermsAndHonours), 1);
    EXPECT_EQ(yaku1[Yaku::kAllTermsAndHonours], 2);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,s1,s1,ew,ew,ew,rd,rd,rd").Tsumo("m9")));
    EXPECT_EQ(yaku2.count(Yaku::kAllTermsAndHonours), 0);
}

TEST_F(YakuTest, HalfFlush)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,ew,ew,ew,rd,rd").Tsumo("m6")));
    EXPECT_EQ(yaku1.count(Yaku::kHalfFlush), 1);
    EXPECT_EQ(yaku1[Yaku::kHalfFlush], 3);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,rd,rd").Pon("ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kHalfFlush), 1);
    EXPECT_EQ(yaku2[Yaku::kHalfFlush], 2);

    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,s1,s1,ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku3.count(Yaku::kHalfFlush), 0);

    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8,m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku4.count(Yaku::kHalfFlush), 0);

    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8").Pon("m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku5.count(Yaku::kHalfFlush), 0);
}

TEST_F(YakuTest, FullFlush)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,ew,ew,ew,rd,rd").Tsumo("m6")));
    EXPECT_EQ(yaku1.count(Yaku::kFullFlush), 0);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,rd,rd").Pon("ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku2.count(Yaku::kFullFlush), 0);

    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,s1,s1,ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku3.count(Yaku::kFullFlush), 0);

    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8,m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku4.count(Yaku::kFullFlush), 1);
    EXPECT_EQ(yaku4[Yaku::kFullFlush], 6);

    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8").Pon("m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku5.count(Yaku::kFullFlush), 1);
    EXPECT_EQ(yaku5[Yaku::kFullFlush], 5);
}
