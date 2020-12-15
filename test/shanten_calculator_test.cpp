#include "gtest/gtest.h"
#include <mj/shanten_calculator.h>
#include <mj/hand.h>

using namespace mj;

TEST(shanten, normal)
{
    std::array<uint8_t, 34> tiles = Hand(
        Tile::Create({"m1", "m1", "m2", "m3", "m4", "m6", "m6", "m6", "p2", "p2", "p2", "ww", "ww"})).ToArray();

    EXPECT_EQ(ShantenCalculator::ShantenNumber(tiles), 0);
}

TEST(shanten, seven_pairs)
{
    std::array<uint8_t, 34> tiles;
    tiles = Hand(
            Tile::Create({"m1", "m1", "m2", "m2", "p3", "p3", "p7", "p7", "ew", "ew", "sw", "rd", "wd"})
            ).ToArray();
    EXPECT_EQ(ShantenCalculator::ShantenSevenPairs(tiles), 1);

    tiles = Hand(
            Tile::Create({"m1", "m1", "m2", "m2", "p3", "p3", "p3", "p3", "ew", "ew", "sw", "rd", "wd"})
    ).ToArray();
    EXPECT_EQ(ShantenCalculator::ShantenSevenPairs(tiles), 2);
}

TEST(shanten, thirteen_orphan)
{
    std::array<uint8_t, 34> tiles;
    tiles = Hand(
            Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "p4"})
    ).ToArray();
    EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles), 1);

    tiles = Hand(
            Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "wd", "p4"})
    ).ToArray();
    EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles), 1);

    tiles = Hand(
            Tile::Create({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "ww", "wd", "wd", "p4"})
    ).ToArray();
    EXPECT_EQ(ShantenCalculator::ShantenThirteenOrphans(tiles), 2);
}