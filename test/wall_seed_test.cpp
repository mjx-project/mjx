#include "gtest/gtest.h"
#include "wall_seed.h"

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

