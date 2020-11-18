#include "gtest/gtest.h"
#include <mj/wall_seed.h>
#include <mj/state.h>

using namespace mj;

TEST(WallSeedTest, constructor) {
    auto seeds = WallSeed();
    std::set<std::uint64_t> st;
    const int ROUND = 10, HONBA = 10;
    for (int i = 0; i < 100; ++i) {
        for (int r = 0; r < ROUND; ++r) {
            for (int h = 0; h < HONBA; ++h) {
                st.insert(seeds.Get(r, h));
            }
        }
    }
    EXPECT_EQ(st.size(), ROUND * HONBA);
}

TEST(WallSeedTest, fixed_seed) {
    auto seeds1 = WallSeed(9999);
    auto seeds2 = WallSeed(9999);
    const int ROUND = 10, HONBA = 10;
    for (int r = 0; r < ROUND; ++r) {
        for (int h = 0; h < HONBA; ++h) {
            EXPECT_EQ(seeds1.Get(r, h), seeds2.Get(r, h));
        }
    }
}
