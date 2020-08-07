#include "gtest/gtest.h"
#include "types.h"

using namespace mj;


TEST(types, ToRelativePos) {
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitEast, AbsolutePos::kInitEast), RelativePos::kSelf);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitEast, AbsolutePos::kInitSouth), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitEast, AbsolutePos::kInitWest), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitEast, AbsolutePos::kInitNorth), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitEast), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitSouth), RelativePos::kSelf);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitWest), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitNorth), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitWest, AbsolutePos::kInitEast), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitWest, AbsolutePos::kInitSouth), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitWest, AbsolutePos::kInitWest), RelativePos::kSelf);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitWest, AbsolutePos::kInitNorth), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitEast), RelativePos::kRight);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitSouth), RelativePos::kMid);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitWest), RelativePos::kLeft);
    EXPECT_EQ(ToRelativePos(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitNorth), RelativePos::kSelf);
}

TEST(types, ToSeatWind) {
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitEast, AbsolutePos::kInitEast), Wind::kEast);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitEast, AbsolutePos::kInitSouth), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitEast, AbsolutePos::kInitWest), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitEast, AbsolutePos::kInitNorth), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitEast), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitSouth), Wind::kEast);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitWest), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitSouth, AbsolutePos::kInitNorth), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitWest, AbsolutePos::kInitEast), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitWest, AbsolutePos::kInitSouth), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitWest, AbsolutePos::kInitWest), Wind::kEast);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitWest, AbsolutePos::kInitNorth), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitEast), Wind::kNorth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitSouth), Wind::kWest);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitWest), Wind::kSouth);
    EXPECT_EQ(ToSeatWind(mj::AbsolutePos::kInitNorth, AbsolutePos::kInitNorth), Wind::kEast);
}

