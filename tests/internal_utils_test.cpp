#include <mjx/internal/types.h>
#include <mjx/internal/utils.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_utils, Any) {
  auto target = TileType::kRD;

  // initializer list
  EXPECT_TRUE(Any(target, {TileType::kWD, TileType::kGD, TileType::kRD}));
  EXPECT_FALSE(Any(
      target, {TileType::kEW, TileType::kNW, TileType::kWW, TileType::kNW}));
  // vector
  EXPECT_TRUE(
      Any(target, std::vector({TileType::kWD, TileType::kGD, TileType::kRD})));
  EXPECT_FALSE(Any(target, std::vector({TileType::kEW, TileType::kNW,
                                        TileType::kWW, TileType::kNW})));
}
