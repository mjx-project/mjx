#include <numeric>
#include "gtest/gtest.h"
#include "consts.h"
#include "win_cache.h"
#include "yaku_evaluator.h"

using namespace mj;

auto win_cache = WinningHandCache();
auto yaku_evaluator = YakuEvaluator(win_cache);

TEST(YakuEvaluator, kBitFullyConcealedHand) // 門前清自摸和
{
    auto hand = Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    hand.Tsumo(Tile("m1", 1));
    auto yakus = yaku_evaluator.Apply(hand);
    EXPECT_TRUE(std::find(yakus.begin(), yakus.end(), Yaku::kFullyConcealedHand) != yakus.end());

    hand = Hand({"m1", "m9", "p1", "p9", "s1", "s9", "ew", "sw", "ww", "nw", "wd", "gd", "rd"});
    hand.Ron(Tile("m1", 1));
    yakus = yaku_evaluator.Apply(hand);
    EXPECT_FALSE(std::find(yakus.begin(), yakus.end(), Yaku::kFullyConcealedHand) != yakus.end());
}

// kBitFullyConcealedHand         // 門前清自摸和
// kBitRiichi                     // 立直
// kBitIppatsu                    // 一発
// kBitRobbingKan                 // 槍槓
// kBitAfterKan                   // 嶺上開花
// kBitBottomOfTheSea             // 海底摸月
// kBitBottomOfTheRiver           // 河底撈魚
// kBitPinfu                      // 平和
// kBitAllSimples                 // 断幺九
// kBitPureDoubleChis             // 一盃口
// kBitSeatWindEast               // 自風 東
// kBitSeatWindSouth              // 自風 南
// kBitSeatWindWest               // 自風 西
// kBitSeatWindNorth              // 自風 北
// kBitPrevalentWindEast          // 場風 東
// kBitPrevalentWindSouth         // 場風 南
// kBitPrevalentWindWest          // 場風 西
// kBitPrevalentWindNorth         // 場風 北
// kBitWhiteDragon                // 役牌 白
// kBitGreenDragon                // 役牌 發
// kBitRedDragon                  // 役牌 中
// kBitDoubleRiichi               // 両立直
// kBitSevenPairs                 // 七対子
// kBitOutsideHand                // 混全帯幺九
// kBitPureStraight               // 一気通貫
// kBitMixedTripleChis            // 三色同順
// kBitTriplePons                 // 三色同刻
// kBitThreeKans                  // 三槓子
// kBitAllPons                    // 対々和
// kBitThreeConcealedPons         // 三暗刻
// kBitLittleThreeDragons         // 小三元
// kBitAllTermsAndHonours         // 混老頭
// kBitTwicePureDoubleChis        // 二盃口
// kBitTerminalsInAllSets         // 純全帯幺九
// kBitHalfFlush                  // 混一色
// kBitFullFlush                  // 清一色
// kBitBlessingOfMan              // 人和
// kBitBlessingOfHeaven           // 天和
// kBitBlessingOfEarth            // 地和
// kBitBigThreeDragons            // 大三元
// kBitFourConcealedPons          // 四暗刻
// kBitCompletedFourConcealedPons // 四暗刻単騎
// kBitAllHonours                 // 字一色
// kBitAllGreen                   // 緑一色
// kBitAllTerminals               // 清老頭
// kBitNineGates                  // 九蓮宝燈
// kBitPureNineGates              // 純正九蓮宝燈
// kBitThirteenOrphans            // 国士無双
// kBitCompletedThirteenOrphans   // 国士無双１３面
// kBitBigFourWinds               // 大四喜
// kBitLittleFourWinds            // 小四喜
// kBitFourKans                   // 四槓子
// kBitDora                       // ドラ
// kBitReversedDora               // 裏ドラ
// kBitRedDora                    // 赤ドラ
