#include "gtest/gtest.h"
#include "yaku_evaluator.h"

#include "types.h"
#include "hand.h"

class YakuTest : public ::testing::Test {
protected:
    // virtual void SetUp() {}
    // virtual void TearDown() {}

    mj::YakuEvaluator evaluator;
};


TEST_F(YakuTest, Empty)
{
    EXPECT_TRUE(evaluator.Has(mj::Hand(mj::HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"))));
    EXPECT_EQ(evaluator.Eval(mj::Hand(mj::HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"))).size(), 0);
}

TEST_F(YakuTest, FullyConcealdHand)
{
    auto yaku1 = evaluator.Eval(
            mj::Hand(mj::HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(std::count(yaku1.begin(), yaku1.end(), mj::Yaku::kFullyConcealedHand), 1);

    auto yaku2 = evaluator.Eval(
            mj::Hand(mj::HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9,p1,p1,p1").Ron("m6")));
    EXPECT_EQ(std::count(yaku2.begin(), yaku2.end(), mj::Yaku::kFullyConcealedHand), 0);

    auto yaku3 = evaluator.Eval(
            mj::Hand(mj::HandParams("m1,m2,m3,m4,m5,rd,rd,m7,m8,m9").Pon("p1,p1,p1").Tsumo("m6")));
    EXPECT_EQ(std::count(yaku3.begin(), yaku3.end(), mj::Yaku::kFullyConcealedHand), 0);
}
