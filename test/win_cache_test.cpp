#include "gtest/gtest.h"
#include "win_cache.h"


TEST(win_cache, Yaku)
{
    mj::WinningHandCache win_cache;
    EXPECT_TRUE(win_cache.Has("2,111,111111111"));
    EXPECT_FALSE(win_cache.Has("2,3,2,1111111"));
}
