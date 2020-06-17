#include <numeric>
#include "gtest/gtest.h"
#include "consts.h"
#include "win_cache.h"
#include "yaku_evaluator.h"

using namespace mj;

auto win_cache = WinningHandCache();
auto yaku_evaluator = YakuEvaluator(win_cache);

TEST(yaku_evaluator, FullyConcealedHand) // 門前清自摸和
{
    auto hand = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd").Tsumo("m1"));
    auto yakus = yaku_evaluator.Apply(hand);
    EXPECT_TRUE(std::find(yakus.begin(), yakus.end(), Yaku::kFullyConcealedHand) != yakus.end());

    hand = Hand(HandParams("m1,m9,p1,p9,s1,s9,ew,sw,ww,nw,wd,gd,rd").Ron("m1"));
    yakus = yaku_evaluator.Apply(hand);
    EXPECT_FALSE(std::find(yakus.begin(), yakus.end(), Yaku::kFullyConcealedHand) != yakus.end());
}

// FullyConcealedHand         門前清自摸和
// Riichi                     立直
// Ippatsu                    一発
// RobbingKan                 槍槓
// AfterKan                   嶺上開花
// BottomOfTheSea             海底摸月
// BottomOfTheRiver           河底撈魚
// Pinfu                      平和
// AllSimples                 断幺九
// PureDoubleChis             一盃口
// SeatWindEast               自風 東
// SeatWindSouth              自風 南
// SeatWindWest               自風 西
// SeatWindNorth              自風 北
// PrevalentWindEast          場風 東
// PrevalentWindSouth         場風 南
// PrevalentWindWest          場風 西
// PrevalentWindNorth         場風 北
// WhiteDragon                役牌 白
// GreenDragon                役牌 發
// RedDragon                  役牌 中
// DoubleRiichi               両立直
// SevenPairs                 七対子
// OutsideHand                混全帯幺九
// PureStraight               一気通貫
// MixedTripleChis            三色同順
// TriplePons                 三色同刻
// ThreeKans                  三槓子
// AllPons                    対々和
// ThreeConcealedPons         三暗刻
// LittleThreeDragons         小三元
// AllTermsAndHonours         混老頭
// TwicePureDoubleChis        二盃口
// TerminalsInAllSets         純全帯幺九
// HalfFlush                  混一色
// FullFlush                  清一色
// BlessingOfMan              人和
// BlessingOfHeaven           天和
// BlessingOfEarth            地和
// BigThreeDragons            大三元
// FourConcealedPons          四暗刻
// CompletedFourConcealedPons 四暗刻単騎
// AllHonours                 字一色
// AllGreen                   緑一色
// AllTerminals               清老頭
// NineGates                  九蓮宝燈
// PureNineGates              純正九蓮宝燈
// ThirteenOrphans            国士無双
// CompletedThirteenOrphans   国士無双１３面
// BigFourWinds               大四喜
// LittleFourWinds            小四喜
// FourKans                   四槓子
// Dora                       ドラ
// ReversedDora               裏ドラ
// RedDora                    赤ドラ
