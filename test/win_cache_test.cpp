#include "gtest/gtest.h"
#include "win_cache.h"


TEST(win_cache, Yaku)
{
    const auto &win_cache = mj::WinHandCache::instance();
    EXPECT_TRUE(win_cache.Has("2,111,111111111"));
    EXPECT_FALSE(win_cache.Has("2,3,2,1111111"));
    EXPECT_TRUE(win_cache.Has("2,1,1,1,1,1,1,1,1,1,1,1,1"));
}
