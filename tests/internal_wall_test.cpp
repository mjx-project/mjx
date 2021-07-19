#include <mjx/internal/wall.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_wall, initial_hand) {
  // From https://tenhou.net/0/?log=2011020417gm-00a9-0000-b67fcaa3&tw=1
  auto tiles = Tile::Create(
      {48,  16,  19,  34,  17,  62,  79, 52,  55,  30,  12,  26,  120, 130,
       42,  67,  2,   76,  13,  7,   56, 57,  82,  98,  31,  90,  3,   4,
       114, 93,  5,   61,  128, 1,   39, 121, 32,  103, 24,  70,  80,  125,
       66,  102, 20,  108, 41,  100, 87, 54,  78,  84,  107, 47,  14,  131,
       96,  51,  68,  85,  28,  10,  6,  18,  122, 49,  134, 109, 116, 127,
       105, 65,  92,  101, 29,  23,  83, 115, 77,  38,  15,  43,  94,  21,
       50,  91,  89,  45,  97,  37,  25, 35,  60,  132, 119, 135, 59,  0,
       9,   27,  53,  58,  118, 110, 22, 124, 69,  44,  33,  8,   74,  129,
       64,  88,  72,  75,  104, 73,  71, 81,  111, 86,  36,  99,  133, 11,
       40,  113, 123, 95,  112, 117, 46, 126, 63,  106});
  auto wall = Wall(0, tiles);
  EXPECT_EQ(wall.initial_hand_tiles(AbsolutePos::kInitEast),
            Tile::Create({48, 16, 19, 34, 2, 76, 13, 7, 128, 1, 39, 121, 87}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitSouth),
      Tile::Create({17, 62, 79, 52, 56, 57, 82, 98, 32, 103, 24, 70, 54}));
  EXPECT_EQ(wall.initial_hand_tiles(AbsolutePos::kInitWest),
            Tile::Create({55, 30, 12, 26, 31, 90, 3, 4, 80, 125, 66, 102, 78}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitNorth),
      Tile::Create({120, 130, 42, 67, 114, 93, 5, 61, 20, 108, 41, 100, 84}));
  tiles = Tile::Create(
      {117, 42,  114, 28,  70,  124, 97,  56,  5,   32,  81, 46,  52,  41,
       105, 21,  80,  87,  73,  2,   33,  71,  13,  118, 7,  119, 129, 116,
       83,  40,  17,  89,  31,  27,  68,  25,  24,  86,  90, 101, 104, 103,
       30,  130, 50,  4,   11,  60,  47,  34,  3,   120, 62, 59,  113, 82,
       22,  108, 100, 43,  132, 79,  88,  94,  12,  63,  84, 38,  107, 131,
       111, 77,  95,  109, 8,   106, 61,  16,  75,  96,  6,  58,  133, 125,
       102, 98,  23,  19,  36,  14,  91,  69,  37,  44,  78, 127, 54,  122,
       51,  76,  0,   10,  135, 39,  121, 134, 93,  64,  85, 35,  9,   45,
       67,  18,  74,  128, 115, 48,  110, 26,  65,  112, 29, 20,  66,  49,
       1,   15,  55,  53,  72,  99,  92,  126, 123, 57});
  wall = Wall(5, tiles);
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitEast),
      Tile::Create({52, 41, 105, 21, 83, 40, 17, 89, 50, 4, 11, 60, 120}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitSouth),
      Tile::Create({117, 42, 114, 28, 80, 87, 73, 2, 31, 27, 68, 25, 47}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitWest),
      Tile::Create({70, 124, 97, 56, 33, 71, 13, 118, 24, 86, 90, 101, 34}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitNorth),
      Tile::Create({5, 32, 81, 46, 7, 119, 129, 116, 104, 103, 30, 130, 3}));
  tiles = Tile::Create(
      {61,  132, 36,  46,  19,  55,  77,  86,  12,  31,  25,  22,  40,  7,
       81,  21,  129, 89,  14,  59,  131, 51,  135, 33,  38,  26,  8,   1,
       111, 130, 62,  18,  101, 16,  127, 123, 119, 11,  64,  97,  122, 94,
       52,  39,  98,  93,  71,  103, 47,  110, 74,  82,  107, 87,  35,  88,
       42,  2,   48,  37,  73,  79,  5,   17,  75,  134, 45,  90,  23,  30,
       96,  104, 117, 58,  126, 78,  83,  120, 102, 50,  92,  100, 4,   28,
       76,  24,  121, 69,  27,  72,  63,  112, 43,  133, 114, 54,  70,  13,
       95,  85,  116, 67,  3,   34,  57,  105, 60,  10,  41,  91,  15,  109,
       115, 6,   84,  20,  65,  118, 125, 66,  32,  128, 108, 53,  44,  68,
       80,  113, 56,  106, 99,  124, 49,  0,   9,   29});
  wall = Wall(7, tiles);
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitEast),
      Tile::Create({19, 55, 77, 86, 131, 51, 135, 33, 119, 11, 64, 97, 110}));
  EXPECT_EQ(wall.initial_hand_tiles(AbsolutePos::kInitSouth),
            Tile::Create({12, 31, 25, 22, 38, 26, 8, 1, 122, 94, 52, 39, 74}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitWest),
      Tile::Create({40, 7, 81, 21, 111, 130, 62, 18, 98, 93, 71, 103, 82}));
  EXPECT_EQ(
      wall.initial_hand_tiles(AbsolutePos::kInitNorth),
      Tile::Create({61, 132, 36, 46, 129, 89, 14, 59, 101, 16, 127, 123, 47}));
}

TEST(internal_wall, Draw) {
  // カンなしで70回ツモが存在する
  auto wall = Wall(0, 0, 9999);
  for (int i = 0; i < 70; ++i) {
    EXPECT_TRUE(wall.HasDrawLeft());
    wall.Draw();
  }
  EXPECT_FALSE(wall.HasDrawLeft());
}

TEST(internal_wall, KanDraw) {
  // カンがあると、その分ツモ数が減る
  auto wall = Wall(0, 0, 9999);
  for (int i = 0; i < 35; ++i) {
    EXPECT_TRUE(wall.HasDrawLeft());
    wall.Draw();
  }
  for (int i = 0; i < 4; ++i) {
    wall.KanDraw();
    wall.AddKanDora();
  }
  for (int i = 0; i < 31; ++i) {
    EXPECT_TRUE(wall.HasDrawLeft());
    wall.Draw();
  }
  EXPECT_FALSE(wall.HasDrawLeft());

  // From https://tenhou.net/0/?log=2010112714gm-00a9-0000-d497e395
  auto tiles = Tile::Create(
      {24, 48,  11,  122, 135, 75,  128, 65,  111, 13, 46,  131, 120, 125,
       71, 107, 118, 54,  52,  121, 109, 26,  42,  93, 119, 123, 21,  8,
       80, 0,   116, 58,  132, 50,  64,  133, 63,  32, 35,  117, 100, 34,
       82, 83,  74,  87,  89,  72,  29,  1,   67,  56, 78,  103, 124, 126,
       59, 69,  38,  112, 61,  22,  25,  60,  86,  31, 99,  127, 23,  68,
       3,  5,   33,  51,  15,  73,  43,  18,  85,  55, 53,  92,  114, 129,
       97, 113, 40,  115, 90,  81,  4,   2,   19,  45, 105, 6,   36,  10,
       94, 91,  12,  28,  17,  30,  106, 101, 79,  47, 49,  102, 104, 95,
       41, 20,  16,  110, 96,  14,  134, 37,  108, 70, 130, 27,  39,  7,
       57, 84,  44,  77,  76,  66,  98,  9,   88,  62});
  wall = Wall(2, tiles);
  EXPECT_EQ(wall.KanDraw(), Tile(88));

  // TODO(sotestuk): add cases for 4 KANs from Tenhou
}

TEST(internal_wall, AddKanDra) {
  auto wall = Wall(0, 0, 9999);
  wall.KanDraw();
  auto [kan_dora_ind1, ura_kan_dora_ind1] = wall.AddKanDora();
  EXPECT_EQ(kan_dora_ind1, wall.dora_indicators().back());
  EXPECT_EQ(ura_kan_dora_ind1, wall.ura_dora_indicators().back());
  auto [kan_dora_ind2, ura_kan_dora_ind2] = wall.AddKanDora();
  EXPECT_EQ(kan_dora_ind2, wall.dora_indicators().back());
  EXPECT_EQ(ura_kan_dora_ind2, wall.ura_dora_indicators().back());
}

TEST(internal_wall, doras) {
  auto wall = Wall(0, 0, 9999);
  EXPECT_EQ(wall.dora_indicators().size(), 1);
  EXPECT_EQ(wall.ura_dora_indicators().size(), 1);
  for (int i = 0; i < 4; ++i) {
    wall.KanDraw();
    wall.AddKanDora();
  }
  EXPECT_EQ(wall.dora_indicators().size(), 5);
  EXPECT_EQ(wall.ura_dora_indicators().size(), 5);

  // From https://tenhou.net/0/?log=2010112714gm-00a9-0000-d497e395
  auto tiles = Tile::Create(
      {24, 48,  11,  122, 135, 75,  128, 65,  111, 13, 46,  131, 120, 125,
       71, 107, 118, 54,  52,  121, 109, 26,  42,  93, 119, 123, 21,  8,
       80, 0,   116, 58,  132, 50,  64,  133, 63,  32, 35,  117, 100, 34,
       82, 83,  74,  87,  89,  72,  29,  1,   67,  56, 78,  103, 124, 126,
       59, 69,  38,  112, 61,  22,  25,  60,  86,  31, 99,  127, 23,  68,
       3,  5,   33,  51,  15,  73,  43,  18,  85,  55, 53,  92,  114, 129,
       97, 113, 40,  115, 90,  81,  4,   2,   19,  45, 105, 6,   36,  10,
       94, 91,  12,  28,  17,  30,  106, 101, 79,  47, 49,  102, 104, 95,
       41, 20,  16,  110, 96,  14,  134, 37,  108, 70, 130, 27,  39,  7,
       57, 84,  44,  77,  76,  66,  98,  9,   88,  62});
  wall = Wall(2, tiles);
  wall.KanDraw();
  wall.AddKanDora();
  EXPECT_EQ(wall.dora_indicators(), Tile::Create({76, 44}));
}

TEST(internal_wall, ura_doras) {
  // From https://tenhou.net/0/?log=2010112714gm-00a9-0000-d497e395
  auto tiles = Tile::Create(
      {24, 48,  11,  122, 135, 75,  128, 65,  111, 13, 46,  131, 120, 125,
       71, 107, 118, 54,  52,  121, 109, 26,  42,  93, 119, 123, 21,  8,
       80, 0,   116, 58,  132, 50,  64,  133, 63,  32, 35,  117, 100, 34,
       82, 83,  74,  87,  89,  72,  29,  1,   67,  56, 78,  103, 124, 126,
       59, 69,  38,  112, 61,  22,  25,  60,  86,  31, 99,  127, 23,  68,
       3,  5,   33,  51,  15,  73,  43,  18,  85,  55, 53,  92,  114, 129,
       97, 113, 40,  115, 90,  81,  4,   2,   19,  45, 105, 6,   36,  10,
       94, 91,  12,  28,  17,  30,  106, 101, 79,  47, 49,  102, 104, 95,
       41, 20,  16,  110, 96,  14,  134, 37,  108, 70, 130, 27,  39,  7,
       57, 84,  44,  77,  76,  66,  98,  9,   88,  62});
  auto wall = Wall(2, tiles);
  wall.KanDraw();
  wall.AddKanDora();
  EXPECT_EQ(wall.ura_dora_indicators(), Tile::Create({66, 77}));
}

TEST(internal_wall, wall_seed_constructor) {
  const int kROUND = 5, kHONBA = 5, kSEED = 1234;
  for (int r = 0; r < kROUND; ++r) {
    for (int h = 0; h < kHONBA; ++h) {
      auto wall1 = Wall(r, h, kSEED).tiles();
      auto wall2 = Wall(r, h, kSEED).tiles();
      for (int i = 0; i < wall1.size(); ++i) {
        EXPECT_EQ(wall1[i], wall2[i]);
      }
    }
  }
}

TEST(internal_wall, WallEqualityOverDevices) {
  auto wall = Wall(0, 0, 9999);
  auto tiles = wall.tiles();
  ASSERT_EQ(tiles.at(0), Tile(42));
  ASSERT_EQ(tiles.at(1), Tile(38));
  ASSERT_EQ(tiles.at(2), Tile(111));
  wall = Wall(4, 4, 1234);
  tiles = wall.tiles();
  ASSERT_EQ(tiles.at(0), Tile(121));
  ASSERT_EQ(tiles.at(1), Tile(113));
  ASSERT_EQ(tiles.at(2), Tile(100));
}
