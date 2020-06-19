#include "gtest/gtest.h"
#include "yaku_evaluator.h"

#include "hand.h"


TEST(yaku_eval, Empty)
{
    mj::YakuEvaluator evaluator;
    EXPECT_TRUE(evaluator.Has(mj::Hand(mj::HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"))));
    EXPECT_EQ(evaluator.Eval(mj::Hand(mj::HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd"))).size(), 0);
}

