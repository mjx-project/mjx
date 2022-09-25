#include <mjx/internal/win_score.h>

#include <optional>

#include "gtest/gtest.h"

using namespace mjx::internal;

TEST(internal_win_score, dealer_tsumo) {
  WinScore score;
  score.AddYakuman(Yaku::kBigThreeDragons);
  score.AddYakuman(Yaku::kAllHonours);

  EXPECT_TRUE(score.HasYakuman(Yaku::kBigThreeDragons));
  EXPECT_TRUE(score.HasYakuman(Yaku::kAllHonours));

  // 親のダブル役満
  auto ten_moves =
      score.TenMoves(AbsolutePos::kInitEast, AbsolutePos::kInitEast);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitEast], 96000);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitSouth], -48000 * 2 / 3);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitWest], -48000 * 2 / 3);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitNorth], -48000 * 2 / 3);
}

TEST(internal_win_score, dealer_ron) {
  WinScore score;
  score.AddYaku(Yaku::kSevenPairs, 2);
  score.AddYaku(Yaku::kRiichi, 1);
  score.set_fu(25);

  EXPECT_EQ(score.HasYaku(Yaku::kSevenPairs), std::make_optional(2));
  EXPECT_EQ(score.HasYaku(Yaku::kRiichi), std::make_optional(1));
  EXPECT_EQ(score.total_fan(), 3);

  auto ten_moves = score.TenMoves(
      AbsolutePos::kInitEast, AbsolutePos::kInitEast, AbsolutePos::kInitSouth);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitEast], 4800);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitSouth], -4800);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitWest], 0);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitNorth], 0);
}

TEST(internal_win_score, non_dealer_tsumo) {
  WinScore score;
  score.AddYaku(Yaku::kRiichi, 1);
  score.AddYaku(Yaku::kIppatsu, 1);
  score.AddYaku(Yaku::kFullyConcealedHand, 1);

  EXPECT_EQ(score.HasYaku(Yaku::kRiichi), std::make_optional(1));
  EXPECT_EQ(score.HasYaku(Yaku::kIppatsu), std::make_optional(1));
  EXPECT_EQ(score.HasYaku(Yaku::kFullyConcealedHand), std::make_optional(1));
  EXPECT_EQ(score.total_fan(), 3);
  score.set_fu(30);

  auto ten_moves =
      score.TenMoves(AbsolutePos::kInitEast, AbsolutePos::kInitSouth);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitEast], 4000);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitSouth], -2000);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitWest], -1000);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitNorth], -1000);
}

TEST(internal_win_score, non_dealer_ron) {
  WinScore score;
  score.AddYaku(Yaku::kAllPons, 2);
  score.AddYaku(Yaku::kHalfFlush, 2);
  score.AddYaku(Yaku::kPrevalentWindEast, 1);
  score.AddYaku(Yaku::kSeatWindSouth, 1);
  score.AddYaku(Yaku::kDora, 3);

  EXPECT_EQ(score.HasYaku(Yaku::kAllPons), std::make_optional(2));
  EXPECT_EQ(score.HasYaku(Yaku::kHalfFlush), std::make_optional(2));
  EXPECT_EQ(score.HasYaku(Yaku::kPrevalentWindEast), std::make_optional(1));
  EXPECT_EQ(score.HasYaku(Yaku::kSeatWindSouth), std::make_optional(1));
  EXPECT_EQ(score.HasYaku(Yaku::kDora), std::make_optional(3));
  EXPECT_EQ(score.total_fan(), 9);

  auto ten_moves = score.TenMoves(
      AbsolutePos::kInitEast, AbsolutePos::kInitSouth, AbsolutePos::kInitWest);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitEast], 16000);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitSouth], 0);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitWest], -16000);
  EXPECT_EQ(ten_moves[AbsolutePos::kInitNorth], 0);
}
