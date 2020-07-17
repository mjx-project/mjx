#include "gtest/gtest.h"

#include "win_score.h"

using namespace mj;

TEST(win_score, dealer_tsumo) {
    WinningScore score;
    score.AddYakuman(Yaku::kBigThreeDragons);
    score.AddYakuman(Yaku::kAllHonours);

    EXPECT_TRUE(score.HasYakuman(Yaku::kBigThreeDragons));
    EXPECT_TRUE(score.HasYakuman(Yaku::kAllHonours));

    auto payment = score.Payment(0, 0, std::nullopt);
    EXPECT_EQ(payment[0], 0);
    EXPECT_EQ(payment[1], 48000*2/3);
    EXPECT_EQ(payment[2], 48000*2/3);
    EXPECT_EQ(payment[3], 48000*2/3);
}

TEST(win_score, dealer_ron) {
    WinningScore score;
    score.AddYaku(Yaku::kSevenPairs, 2);
    score.AddYaku(Yaku::kRiichi, 1);
    score.set_fu(25);

    EXPECT_EQ(score.HasYaku(Yaku::kSevenPairs), std::make_optional(2));
    EXPECT_EQ(score.HasYaku(Yaku::kRiichi), std::make_optional(1));
    EXPECT_EQ(score.total_fan(), 3);

    auto payment = score.Payment(0, 0, 1);
    EXPECT_EQ(payment[0], 0);
    EXPECT_EQ(payment[1], 4800);
    EXPECT_EQ(payment[2], 0);
    EXPECT_EQ(payment[3], 0);
}

TEST(win_score, non_dealer_tsumo) {
    WinningScore score;
    score.AddYaku(Yaku::kRiichi, 1);
    score.AddYaku(Yaku::kIppatsu, 1);
    score.AddYaku(Yaku::kFullyConcealedHand, 1);

    EXPECT_EQ(score.HasYaku(Yaku::kRiichi), std::make_optional(1));
    EXPECT_EQ(score.HasYaku(Yaku::kIppatsu), std::make_optional(1));
    EXPECT_EQ(score.HasYaku(Yaku::kFullyConcealedHand), std::make_optional(1));
    EXPECT_EQ(score.total_fan(), 3);
    score.set_fu(30);

    auto payment = score.Payment(0, 1, std::nullopt);
    EXPECT_EQ(payment[0], 0);
    EXPECT_EQ(payment[1], 2000);
    EXPECT_EQ(payment[2], 1000);
    EXPECT_EQ(payment[3], 1000);
}

TEST(win_score, non_dealer_ron) {
    WinningScore score;
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

    auto payment = score.Payment(0, 1, 2);
    EXPECT_EQ(payment[0], 0);
    EXPECT_EQ(payment[1], 0);
    EXPECT_EQ(payment[2], 16000);
    EXPECT_EQ(payment[3], 0);
}
