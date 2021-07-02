#include <mjx/internal/game_seed.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_game_seed, constructor) {
  auto seeds = GameSeed(9999);
  std::set<std::uint64_t> st;
  const int kROUND = 10, kHONBA = 10;
  for (int i = 0; i < 100; ++i) {
    for (int r = 0; r < kROUND; ++r) {
      for (int h = 0; h < kHONBA; ++h) {
        st.insert(seeds.GetWallSeed(r, h));
      }
    }
  }
  EXPECT_EQ(st.size(), kROUND * kHONBA);
}

TEST(internal_game_seed, fixed_seed) {
  auto seeds1 = GameSeed(9999);
  auto seeds2 = GameSeed(9999);
  const int kROUND = 10, kHONBA = 10;
  for (int r = 0; r < kROUND; ++r) {
    for (int h = 0; h < kHONBA; ++h) {
      EXPECT_EQ(seeds1.GetWallSeed(r, h), seeds2.GetWallSeed(r, h));
    }
  }
}

TEST(internal_game_seed, WallSeedEqualityOverDevice) {
  auto ws = GameSeed(9999);
  EXPECT_EQ(ws.GetWallSeed(0, 0), 7613689384667096742ULL);
  EXPECT_EQ(ws.GetWallSeed(1, 0), 18049619590696111298ULL);
  EXPECT_EQ(ws.GetWallSeed(0, 1), 9100361418872076222ULL);
  // 同じオブジェクトで2回呼び出しても結果は同じ
  EXPECT_EQ(ws.GetWallSeed(0, 0), 7613689384667096742ULL);
  EXPECT_EQ(ws.GetWallSeed(1, 0), 18049619590696111298ULL);
  EXPECT_EQ(ws.GetWallSeed(0, 1), 9100361418872076222ULL);
  // 違うオブジェクトでも結果は同じ
  ws = GameSeed(9999);
  EXPECT_EQ(ws.GetWallSeed(0, 0), 7613689384667096742ULL);
  EXPECT_EQ(ws.GetWallSeed(1, 0), 18049619590696111298ULL);
  EXPECT_EQ(ws.GetWallSeed(0, 1), 9100361418872076222ULL);
}
