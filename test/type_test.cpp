#include "gtest/gtest.h"
#include <mjx/types.h>

using namespace mjx;


TEST(types, ToRelativePos) {
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitEast), RelativePos::kSelf);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitSouth), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitWest), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitNorth), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitEast), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitSouth), RelativePos::kSelf);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitWest), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitNorth), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitEast), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitSouth), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitWest), RelativePos::kSelf);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitNorth), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitEast), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitSouth), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitWest), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitNorth), RelativePos::kSelf);
}

TEST(types, ToSeatWind) {
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitEast), Wind::kEast);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitSouth), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitWest), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitEast, AbsolutePos::kInitNorth), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitEast), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitSouth), Wind::kEast);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitWest), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitSouth, AbsolutePos::kInitNorth), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitEast), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitSouth), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitWest), Wind::kEast);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitWest, AbsolutePos::kInitNorth), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitEast), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitSouth), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitWest), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mjx::AbsolutePos::kInitNorth, AbsolutePos::kInitNorth), Wind::kEast);
}

