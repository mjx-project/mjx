#include "gtest/gtest.h"
#include "wall_seed.h"

using namespace mj;

TEST(WallSeedTest, constructor) {
    auto seeds = WallSeed();
    std::set<std::uint32_t> st;
    const int ROUND = 10, HONBA = 10;
    for (int i = 0; i < WALL_SEED_NUM; ++i) {
        for (int r = 0; r < ROUND; ++r) {
            for (int h = 0; h < HONBA; ++h) {
                st.insert(seeds.Get(i, r, h));
            }
        }
    }
    EXPECT_EQ(st.size(), WALL_SEED_NUM * ROUND * HONBA);
}

