#include <mjx/internal/types.h>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_types, ToRelativePos) {
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitEast, AbsolutePos::kInitEast),
            RelativePos::kSelf);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitEast, AbsolutePos::kInitSouth),
            RelativePos::kRight);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitEast, AbsolutePos::kInitWest),
            RelativePos::kMid);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitEast, AbsolutePos::kInitNorth),
            RelativePos::kLeft);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitSouth, AbsolutePos::kInitEast),
            RelativePos::kLeft);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitSouth, AbsolutePos::kInitSouth),
            RelativePos::kSelf);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitSouth, AbsolutePos::kInitWest),
            RelativePos::kRight);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitSouth, AbsolutePos::kInitNorth),
            RelativePos::kMid);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitWest, AbsolutePos::kInitEast),
            RelativePos::kMid);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitWest, AbsolutePos::kInitSouth),
            RelativePos::kLeft);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitWest, AbsolutePos::kInitWest),
            RelativePos::kSelf);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitWest, AbsolutePos::kInitNorth),
            RelativePos::kRight);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitNorth, AbsolutePos::kInitEast),
            RelativePos::kRight);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitNorth, AbsolutePos::kInitSouth),
            RelativePos::kMid);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitNorth, AbsolutePos::kInitWest),
            RelativePos::kLeft);
  EXPECT_EQ(ToRelativePos(AbsolutePos::kInitNorth, AbsolutePos::kInitNorth),
            RelativePos::kSelf);
}

TEST(internal_types, ToSeatWind) {
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitEast, AbsolutePos::kInitEast),
            Wind::kEast);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitEast, AbsolutePos::kInitSouth),
            Wind::kNorth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitEast, AbsolutePos::kInitWest),
            Wind::kWest);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitEast, AbsolutePos::kInitNorth),
            Wind::kSouth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitSouth, AbsolutePos::kInitEast),
            Wind::kSouth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitSouth, AbsolutePos::kInitSouth),
            Wind::kEast);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitSouth, AbsolutePos::kInitWest),
            Wind::kNorth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitSouth, AbsolutePos::kInitNorth),
            Wind::kWest);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitWest, AbsolutePos::kInitEast),
            Wind::kWest);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitWest, AbsolutePos::kInitSouth),
            Wind::kSouth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitWest, AbsolutePos::kInitWest),
            Wind::kEast);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitWest, AbsolutePos::kInitNorth),
            Wind::kNorth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitNorth, AbsolutePos::kInitEast),
            Wind::kNorth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitNorth, AbsolutePos::kInitSouth),
            Wind::kWest);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitNorth, AbsolutePos::kInitWest),
            Wind::kSouth);
  EXPECT_EQ(ToSeatWind(AbsolutePos::kInitNorth, AbsolutePos::kInitNorth),
            Wind::kEast);
}
