#include <mjx/internal/hand.h>
#include <mjx/internal/shanten_calculator.h>

#include <fstream>
#include <sstream>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(shanten, normal) {
  std::array<uint8_t, 34> tiles =
      Hand(Tile::Create({"m1", "m1", "m2", "m3", "m4", "m6", "m6", "m6", "p2",
                         "p2", "p2", "ww", "ww"}))
          .ToArray();

  EXPECT_EQ(ShantenCalculator::ShantenNumber(tiles), 0);
}

TEST(shanten, thirteen_orphan) {
  std::array<uint8_t, 34> tiles;
  tiles = Hand(Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw",
                             "ww", "nw", "wd", "gd", "p4"}))
              .ToArray();
  EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles), 1);

  tiles = Hand(Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw",
                             "ww", "nw", "wd", "wd", "p4"}))
              .ToArray();
  EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles), 1);

  tiles = Hand(Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw",
                             "ww", "ww", "wd", "wd", "p4"}))
              .ToArray();
  EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles), 2);
}

TEST(shanten, seven_pairs) {
  std::array<uint8_t, 34> tiles;
  tiles = Hand(Tile::Create({"m1", "m1", "m2", "m2", "p3", "p3", "p7", "p7",
                             "ew", "ew", "sw", "rd", "wd"}))
              .ToArray();
  EXPECT_EQ(ShantenCalculator::ShantenSevenPairs(tiles), 1);

  tiles = Hand(Tile::Create({"m1", "m1", "m2", "m2", "p3", "p3", "p3", "p3",
                             "ew", "ew", "sw", "rd", "wd"}))
              .ToArray();
  EXPECT_EQ(ShantenCalculator::ShantenSevenPairs(tiles), 2);
}

TEST(shanten, many_cases) {
  // 事前にロードする
  {
    clock_t start = clock();
    ShantenCalculator::shanten_cache();
    clock_t end = clock();
    const double time =
        static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    std::cout << "loading takes time " << time << "[ms]" << std::endl;
  }

  for (const std::string& testcase :
       {std::string(TEST_RESOURCES_DIR) + "/shanten_testcases/p_hon_10000.txt",
        std::string(TEST_RESOURCES_DIR) + "/shanten_testcases/p_koku_10000.txt",
        std::string(TEST_RESOURCES_DIR) +
            "/shanten_testcases/p_normal_10000.txt",
        std::string(TEST_RESOURCES_DIR) +
            "/shanten_testcases/p_tin_10000.txt"}) {
    clock_t start = clock();

    std::ifstream ifs(testcase, std::ios::in);
    std::string line;

    while (std::getline(ifs, line)) {
      std::array<uint8_t, 34> tiles{};
      std::fill(tiles.begin(), tiles.end(), 0);
      std::stringstream ss(line);
      for (int i = 0; i < 14; ++i) {
        int tile_type;
        ss >> tile_type;
        ++tiles[tile_type];
      }

      int normal, thirteen_orphans, seven_pairs;
      ss >> normal >> thirteen_orphans >> seven_pairs;
      EXPECT_EQ(ShantenCalculator::ShantenNormal(tiles), normal);
      EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles),
                thirteen_orphans);
      EXPECT_EQ(ShantenCalculator::ShantenSevenPairs(tiles), seven_pairs);
    }

    clock_t end = clock();

    const double time =
        static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
    std::cout << testcase << " takes time " << time << "[ms]" << std::endl;
  }
}
