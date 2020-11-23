#include "gtest/gtest.h"
#include <mj/game_seed.h>

using namespace mj;

TEST(WallSeedTest, constructor) {
    auto seeds = WallSeed(9999);
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


TEST(WallSeedTest, WallSeedEqualityOverDevice) {
    auto ws = WallSeed(9999);
    EXPECT_EQ(ws.Get(0, 0), 7613689384667096742ULL);
    EXPECT_EQ(ws.Get(1, 0), 18049619590696111298ULL);
    EXPECT_EQ(ws.Get(0, 1), 9100361418872076222ULL);
}
