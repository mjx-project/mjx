#include "gtest/gtest.h"
#include "types.h"
#include "utils.h"

using namespace mj;

TEST(utils, any_of) {
    auto target = TileType::kRD;

    // initializer list
    EXPECT_TRUE(any_of(target, {TileType::kWD, TileType::kGD, TileType::kRD}));
    EXPECT_FALSE(any_of(target, {TileType::kEW, TileType::kNW, TileType::kWW, TileType::kNW}));
    // vector
    EXPECT_TRUE(any_of(target, std::vector({TileType::kWD, TileType::kGD, TileType::kRD})));
    EXPECT_FALSE(any_of(target, std::vector({TileType::kEW, TileType::kNW, TileType::kWW, TileType::kNW})));
}

