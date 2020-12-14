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
