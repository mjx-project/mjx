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
    EXPECT_EQ(yaku1.HasYaku(Yaku::kFullyConcealedHand), std::make_optional(1));

    // ロンはダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Ron("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kFullyConcealedHand), std::nullopt);

    // 鳴きはダメ
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9").Pon("p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kFullyConcealedHand), std::nullopt);
}

TEST_F(YakuTest, Pinfu)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kPinfu), std::make_optional(1));

    // 鳴きはダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9").Pon("p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kPinfu), std::nullopt);

    // 刻子はダメ
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,s7,s8,s9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kPinfu), std::nullopt);

    // 役牌の雀頭はダメ
    // TODO: 場風, 自風も弾く
    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,s7,s8,s9,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kPinfu), std::nullopt);

    // リャンメン待ち以外はダメ
    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m6,m9,m9,s7,s8,s9,p1,p2,p3").Tsumo("m5")));
    EXPECT_EQ(yaku5.HasYaku(Yaku::kPinfu), std::nullopt);
    auto yaku6 = evaluator.Eval(
            Hand(HandParams("m1,m2,m4,m5,m6,m9,m9,s7,s8,s9,p1,p2,p3").Tsumo("m3")));
    EXPECT_EQ(yaku6.HasYaku(Yaku::kPinfu), std::nullopt);
    auto yaku7 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,m9,s7,s8,s9,p1,p2,p3").Tsumo("m9")));
    EXPECT_EQ(yaku7.HasYaku(Yaku::kPinfu), std::nullopt);
}

TEST_F(YakuTest, AllSimples) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,p2,p3,p4").Tsumo("m6")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kAllSimples), std::make_optional(1));

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kAllSimples), std::nullopt);
}

TEST_F(YakuTest, Dragon) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,wd,wd,wd").Tsumo("m6")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kWhiteDragon), std::make_optional(1));

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,gd,gd,gd").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kGreenDragon), std::make_optional(1));

    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,rd,rd,rd").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kRedDragon), std::make_optional(1));

    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m2,m3,m4,m5,m7,s2,s2,s6,s7,s8,p1,p2,p3").Tsumo("m6")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kWhiteDragon), std::nullopt);
    EXPECT_EQ(yaku4.HasYaku(Yaku::kGreenDragon), std::nullopt);
    EXPECT_EQ(yaku4.HasYaku(Yaku::kRedDragon), std::nullopt);
}

TEST_F(YakuTest, AllTermsAndHonours)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m9,m9,s1,s1,ew,ew,ew,rd,rd,rd").Tsumo("m9")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kAllTermsAndHonours), std::make_optional(2));

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,s1,s1,ew,ew,ew,rd,rd,rd").Tsumo("m9")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kAllTermsAndHonours), std::nullopt);
}

TEST_F(YakuTest, HalfFlush)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,ew,ew,ew,rd,rd").Tsumo("m6")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kHalfFlush), std::make_optional(3));

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,rd,rd").Pon("ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kHalfFlush), std::make_optional(2));

    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,s1,s1,ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kHalfFlush), std::nullopt);

    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8,m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kHalfFlush), std::nullopt);

    // 清一色とは複合しない
    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8").Pon("m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku5.HasYaku(Yaku::kHalfFlush), std::nullopt);
}

TEST_F(YakuTest, FullFlush)
{
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,ew,ew,ew,rd,rd").Tsumo("m6")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kFullFlush), std::nullopt);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,rd,rd").Pon("ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kFullFlush), std::nullopt);

    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m9,m9,m9,s1,s1,ew,ew,ew").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kFullFlush), std::nullopt);

    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8,m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kFullFlush), std::make_optional(6));

    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m7,m7,m7,m8,m8").Pon("m9,m9,m9").Tsumo("m6")));
    EXPECT_EQ(yaku5.HasYaku(Yaku::kFullFlush), std::make_optional(5));
}

TEST_F(YakuTest, PureDoubleChis) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kPureDoubleChis), std::make_optional(1));

    // 鳴いているとダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s5,s6,ew").Pon("p1,p1,p1").Tsumo("ew")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kPureDoubleChis), std::nullopt);

    // 二盃口とは複合しない
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s4,s5,s5,s6,s6,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kPureDoubleChis), std::nullopt);

    // 一盃口要素無し
    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kPureDoubleChis), std::nullopt);
}

TEST_F(YakuTest, TwicePureDoubleChis) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s4,s5,s5,s6,s6,ew").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kTwicePureDoubleChis), std::make_optional(3));

    // 鳴いているとダメ
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s5,s6,ew").Chi("s4,s5,s6").Tsumo("ew")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kTwicePureDoubleChis), std::nullopt);

    // 二盃口要素無し
    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kTwicePureDoubleChis), std::nullopt);
}

TEST_F(YakuTest, SevenPairs) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m3,m3,m6,m6,s4,s4,s8,s8,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kSevenPairs), std::make_optional(2));

    // 二盃口とは複合しない
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m1,m2,m2,m3,m3,s4,s4,s5,s5,s6,s6,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kSevenPairs), std::nullopt);

    // 七対子要素無し
    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kSevenPairs), std::nullopt);
}

TEST_F(YakuTest, AllPons) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m3,m3,m3,s4,s4,s4,ew,ew,rd,rd").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kAllPons), std::make_optional(2));

    // 鳴いててもOK 喰い下がり無し
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m3,m3,m3,ew,ew,rd,rd").Pon("s4,s4,s4").Tsumo("ew")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kAllPons), std::make_optional(2));

    // 順子が含まれるとNG
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m3,m4,m5,ew,ew,rd,rd").Pon("s4,s4,s4").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kAllPons), std::nullopt);
}

TEST_F(YakuTest, PureStraight) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,m7,m8,m9,ew,ew,rd,rd").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kPureStraight), std::make_optional(2));

    // 鳴いててもOK 喰い下がり1翻
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,ew,ew,rd,rd").Chi("m7,m8,m9").Tsumo("ew")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kPureStraight), std::make_optional(1));

    // 一気通貫要素無し
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kPureStraight), std::nullopt);
}

TEST_F(YakuTest, MixedTripleChis) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s1,s2,s3,p1,p2,p3,ew").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kMixedTripleChis), std::make_optional(2));

    // 鳴いててもOK 喰い下がり1翻
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s1,s2,s3,ew").Chi("p1,p2,p3").Tsumo("ew")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kMixedTripleChis), std::make_optional(1));

    // 三色同順要素無し
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kMixedTripleChis), std::nullopt);
}

TEST_F(YakuTest, TriplePons) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m4,m5,m6,s1,s1,s1,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kTriplePons), std::make_optional(2));

    // 鳴いててもOK 喰い下がり無し
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m4,m5,m6,s1,s1,s1,ew").Pon("p1,p1.p1").Tsumo("ew")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kTriplePons), std::make_optional(2));

    // 三色同刻要素無し
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kTriplePons), std::nullopt);
}

TEST_F(YakuTest, OutsideHand) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,m9,s7,s8,s9,ew,ew,ew,rd").Tsumo("rd")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kOutsideHand), std::make_optional(2));

    // 鳴いててもOK 喰い下がり1翻
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,m9,s7,s8,s9,rd").Pon("ew,ew,ew").Tsumo("rd")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kOutsideHand), std::make_optional(1));

    // 混全帯么九要素無し
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kOutsideHand), std::nullopt);

    // 純全帯幺とは複合しない
    auto yaku4 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,m9,s7,s8,s9,p1,p2,p3,p9").Tsumo("p9")));
    EXPECT_EQ(yaku4.HasYaku(Yaku::kOutsideHand), std::nullopt);

    // 混老頭とは複合しない
    auto yaku5 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m9,m9,s1,s1,ew,ew,ew,rd,rd,rd").Tsumo("m9")));
    EXPECT_EQ(yaku5.HasYaku(Yaku::kOutsideHand), std::nullopt);
}

TEST_F(YakuTest, TerminalsInAllSets) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,m9,s7,s8,s9,p1,p2,p3,p9").Tsumo("p9")));
    EXPECT_EQ(yaku1.HasYaku(Yaku::kTerminalsInAllSets), std::make_optional(3));

    // 鳴いててもOK 喰い下がり2翻
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m9,m9,m9,s7,s8,s9,p9").Chi("p1,p2,p3").Tsumo("p9")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kTerminalsInAllSets), std::make_optional(2));

    // 純全帯幺要素無し
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,m6,s4,s5,s6,p1,p1,p1,ew").Tsumo("ew")));
    EXPECT_EQ(yaku3.HasYaku(Yaku::kTerminalsInAllSets), std::nullopt);
}

TEST_F(YakuTest, ThreeKans) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m5,m5").KanClosed("s1,s1,s1,s1")
                    .KanOpened("p4,p4,p4,p4").KanAdded("wd,wd,wd,wd").Tsumo("m1"))
            );
    EXPECT_EQ(yaku1.HasYaku(Yaku::kThreeKans), std::make_optional(2));

    // 三槓子要素なし
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYaku(Yaku::kThreeKans), std::nullopt);
}

TEST_F(YakuTest, BigThreeDragons) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m5,m5,wd,wd,wd").KanClosed("gd,gd,gd,gd").Pon("rd,rd,rd").Tsumo("m1")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kBigThreeDragons), true);

    // 大三元要素なし
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kBigThreeDragons), false);
}

TEST_F(YakuTest, AllHonours) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("ew,ew,ew,sw,sw,sw,wd")
                .Pon("gd,gd,gd").Pon("rd,rd,rd").Tsumo("wd")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kAllHonours), true);

    // 字一色要素なし
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kAllHonours), false);
}

TEST_F(YakuTest, AllGreen) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("s2,s2,s3,s3,s4,s4,s6")
                         .Pon("s8,s8,s8").Pon("gd,gd,gd").Tsumo("s6")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kAllGreen), true);

    // 緑一色要素なし
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kAllGreen), false);
}

TEST_F(YakuTest, AllTerminals) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m9,m9,m9,s1")
                         .Pon("p1,p1,p1").Pon("p9,p9,p9").Tsumo("s1")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kAllTerminals), true);

    // 緑一色要素なし
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kAllTerminals), false);
}

TEST_F(YakuTest, BigFourWinds) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,ew,ew,ew,sw,sw")
                         .Pon("ww,ww,ww").Pon("nw,nw,nw").Tsumo("sw")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kBigFourWinds), true);

    // 大四喜要素なし
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kBigFourWinds), false);
}

TEST_F(YakuTest, LittleFourWinds) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,ew,ew,ew,sw")
                         .Pon("ww,ww,ww").Pon("nw,nw,nw").Tsumo("sw")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kLittleFourWinds), true);

    // 大四喜とは複合しない
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,ew,ew,ew,sw,sw")
                         .Pon("ww,ww,ww").Pon("nw,nw,nw").Tsumo("sw")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kLittleFourWinds), false);

    // 小四喜要素なし
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYakuman(Yaku::kLittleFourWinds), false);
}

TEST_F(YakuTest, ThirteenOrphans) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m9,s1,s9,p1,p9,ew,sw,ww,nw,wd,gd,gd").Tsumo("rd")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kThirteenOrphans), true);

    // 国士無双十三面待ちとは複合しない
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m9,s1,s9,p1,p9,ew,sw,ww,nw,wd,gd,rd").Tsumo("rd")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kThirteenOrphans), false);

    // 国士無双要素なし
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYakuman(Yaku::kThirteenOrphans), false);
}

TEST_F(YakuTest, CompletedThirteenOrphans) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m9,s1,s9,p1,p9,ew,sw,ww,nw,wd,gd,rd").Tsumo("rd")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kCompletedThirteenOrphans), true);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m9,s1,s9,p1,p9,ew,sw,ww,nw,wd,gd,gd").Tsumo("rd")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kCompletedThirteenOrphans), false);

    // 国士無双要素なし
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYakuman(Yaku::kCompletedThirteenOrphans), false);
}

TEST_F(YakuTest, NineGates) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m2,m2,m3,m4,m5,m6,m7,m8,m9,m9").Tsumo("m9")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kNineGates), true);

    // 純正九蓮宝燈とは複合しない
    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9").Tsumo("m9")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kNineGates), false);

    // 九蓮宝燈とは複合しない
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYakuman(Yaku::kNineGates), false);
}

TEST_F(YakuTest, PureNineGates) {
    auto yaku1 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m2,m3,m4,m5,m6,m7,m8,m9,m9,m9").Tsumo("m9")));
    EXPECT_EQ(yaku1.HasYakuman(Yaku::kPureNineGates), true);

    auto yaku2 = evaluator.Eval(
            Hand(HandParams("m1,m1,m1,m2,m2,m3,m4,m5,m6,m7,m8,m9,m9").Tsumo("m9")));
    EXPECT_EQ(yaku2.HasYakuman(Yaku::kPureNineGates), false);

    // 九蓮宝燈とは複合しない
    auto yaku3 = evaluator.Eval(
            Hand(HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(yaku3.HasYakuman(Yaku::kPureNineGates), false);
}
