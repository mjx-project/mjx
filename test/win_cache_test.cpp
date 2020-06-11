#include <numeric>
#include "gtest/gtest.h"
#include "consts.h"
#include "win_cache.h"

using namespace mj;


TEST(win_cache, Yaku)
{
    auto cache = WinningHandCache();
    EXPECT_EQ(cache.Size(), 3853);
    // fully_concealed_hand, // 門前清自摸和
    // riichi, // 立直
    // ippatsu, // 一発
    // robbing_a_kan, // 槍槓
    // after_a_kan, // 嶺上開花
    // bottom_of_the_sea, // 海底摸月
    // bottom_of_the_river, // 河底撈魚
    // pinfu, // 平和
    EXPECT_TRUE(cache.YakuBit("2,111,111111111") & kBitPinfu);
    EXPECT_FALSE(cache.YakuBit("2,3,111111111") & kBitPinfu);
    // all_simples, // 断幺九
    EXPECT_TRUE(cache.YakuBit("2,111,111,111111") & kBitAllSimples);
    EXPECT_FALSE(cache.YakuBit("2,111,111111111") & kBitAllSimples);
    // pure_double_chis, // 一盃口
    EXPECT_TRUE(cache.YakuBit("2,222222") & kBitAllSimples);
    EXPECT_FALSE(cache.YakuBit("2,2,22,222") & kBitAllSimples);
    // seat_wind_east,// 自風 東
    EXPECT_TRUE(cache.YakuBit("2,3,111111111") & kBitSeatWindEast);
    EXPECT_FALSE(cache.YakuBit("2,111,111111111") & kBitSeatWindEast);
    // seat_wind_south, // 自風 南
    // seat_wind_west, // 自風 西
    // seat_wind_north, // 自風 北
    // prevalent_wind_east, // 場風 東
    // prevalent_wind_south, // 場風 南
    // prevalent_wind_west, // 場風 西
    // prevalent_wind_north, // 場風 北
    // white_dragon, // 役牌 白
    // green_dragon, // 役牌 發
    // red_dragon, // 役牌 中
    // double_riichi, // 両立直
    // seven_pairs, // 七対子
    EXPECT_TRUE(cache.YakuBit("2222222") & kBitSevenPairs);
    EXPECT_FALSE(cache.YakuBit("222422") & kBitSevenPairs);
    // outside_hand, // 混全帯幺九
    EXPECT_TRUE(cache.YakuBit("2,3,3,111,111") & kBitOutsideHand);
    EXPECT_FALSE(cache.YakuBit("2,3,3,111111") & kBitOutsideHand);
    // pure_straight, // 一気通貫
    EXPECT_TRUE(cache.YakuBit("431111111") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("111,431111") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("111,141,1112") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("1413113") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("1314221") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("3,3,1421") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("3,311,11211") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("31113131") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("111,1221131") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("141112112") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("33,112112") & kBitPureStraight);
    EXPECT_FALSE(cache.YakuBit("3,3,111,1112") & kBitPureStraight);
    // // mixed_triple_chis, // 三色同順
    EXPECT_TRUE(cache.YakuBit("2,111,111,114") & kBitMixedTripleChis);
    EXPECT_FALSE(cache.YakuBit("2,33,111,111") & kBitSeatWindEast);
    // triple_pons, // 三色同刻
    EXPECT_TRUE(cache.YakuBit("2,3,3,114") & kBitTriplePons);
    EXPECT_FALSE(cache.YakuBit("2,111,111,114") & kBitTriplePons);
    EXPECT_FALSE(cache.YakuBit("2,3,333") & kBitTriplePons);
    // three_kans, // 三槓子
    EXPECT_TRUE(cache.YakuBit("2,3,3,1113") & kBitThreeKans);
    EXPECT_FALSE(cache.YakuBit("2,3,3,114") & kBitThreeKans);
    // all_pons, // 対々和
    EXPECT_TRUE(cache.YakuBit("2,3,333") & kBitAllPons);
    EXPECT_FALSE(cache.YakuBit("2,111,333") & kBitAllPons);
    // three_concealed_pons, // 三暗刻
    EXPECT_TRUE(cache.YakuBit("2,111,333") & kBitThreeConcealedPons);
    EXPECT_FALSE(cache.YakuBit("2,111,11133") & kBitThreeConcealedPons);
    // little_three_dragons, // 小三元
    EXPECT_TRUE(cache.YakuBit("2,3,3,3,111") & kBitLittleThreeDragons);
    EXPECT_FALSE(cache.YakuBit("2,111,333") & kBitLittleThreeDragons);
    // all_terms_and_honours, // 混老頭
    // EXPECT_TRUE(cache.yaku("2,3,3,3,3") & bit_all_terms_and_honours);
    EXPECT_FALSE(cache.YakuBit("2,3,333") & kBitAllTermsAndHonours);
    // twice_pure_double_chis, // 二盃口
    EXPECT_TRUE(cache.YakuBit("2,222,222") & kBitTwicePureDoubleChis);
    EXPECT_FALSE(cache.YakuBit("2,22,2222") & kBitTwicePureDoubleChis);
    // terminals_in_all_sets, // 純全帯幺九
    EXPECT_TRUE(cache.YakuBit("2,111,111,111,111") & kBitTerminalsInAllSets);
    EXPECT_FALSE(cache.YakuBit("2,111,111,111111") & kBitTerminalsInAllSets);
    EXPECT_FALSE(cache.YakuBit("3,3,4112") & kBitTerminalsInAllSets);
    EXPECT_FALSE(cache.YakuBit("111,2333") & kBitTerminalsInAllSets);
    EXPECT_FALSE(cache.YakuBit("111,111,2114") & kBitTerminalsInAllSets);
    EXPECT_FALSE(cache.YakuBit("3,3,314") & kBitTerminalsInAllSets);
    // half_flush, // 混一色
    EXPECT_TRUE(cache.YakuBit("2,3,111111111") & kBitHalfFlush);
    EXPECT_FALSE(cache.YakuBit("23,111111111") & kBitHalfFlush);
    // full_flush, // 清一色
    EXPECT_TRUE(cache.YakuBit("222111113") & kBitFullFlush);
    EXPECT_FALSE(cache.YakuBit("222,111113") & kBitFullFlush);
    // blessing_of_man, // 人和
    // blessing_of_heaven, // 天和
    // blessing_of_earth, // 地和
    // big_three_dragons, // 大三元
    EXPECT_TRUE(cache.YakuBit("3,3,3,23") & kBitBigThreeDragons);
    EXPECT_FALSE(cache.YakuBit("3,3,233") & kBitBigThreeDragons);
    // // four_concealed_pons, // 四暗刻
    EXPECT_TRUE(cache.YakuBit("23333") & kBitFourConcealedPons);
    EXPECT_FALSE(cache.YakuBit("111,2333") & kBitFourConcealedPons);
    // completed_four_concealed_pons, // 四暗刻単騎
    EXPECT_TRUE(cache.YakuBit("23333") & kBitCompletedFourConcealedPons);
    EXPECT_FALSE(cache.YakuBit("111,2333") & kBitCompletedFourConcealedPons);
    // all_honours, // 字一色
    EXPECT_TRUE(cache.YakuBit("2,3,3,3,3") & kBitAllHonours);
    EXPECT_FALSE(cache.YakuBit("3,3,3,23") & kBitAllHonours);
    // all_green, // 緑一色
    EXPECT_TRUE(cache.YakuBit("2,3,3,222") & kBitAllGreen);
    EXPECT_FALSE(cache.YakuBit("2,33,222") & kBitAllGreen);
    // all_terminals, // 清老頭
    EXPECT_TRUE(cache.YakuBit("2,3,3,3,3") & kBitAllTerminals);
    EXPECT_FALSE(cache.YakuBit("2,3,3,33") & kBitAllTerminals);
    // nine_gates, // 九蓮宝燈
    EXPECT_TRUE(cache.YakuBit("321111113") & kBitNineGates);
    EXPECT_FALSE(cache.YakuBit("222111113") & kBitNineGates);
    // pure_nine_gates, // 純正九蓮宝燈
    EXPECT_TRUE(cache.YakuBit("321111113") & kBitPureNineGates);
    EXPECT_FALSE(cache.YakuBit("222111113") & kBitPureNineGates);
    // thirteen_orphans, // 国士無双
    EXPECT_TRUE(cache.YakuBit("1,1,1,1,1,1,1,1,1,1,1,1,2") & kBitThirteenOrphans);
    EXPECT_FALSE(cache.YakuBit("2,111,111,111,111") & kBitThirteenOrphans);
    // completed_thirteen_orphans, // 国士無双１３面
    EXPECT_TRUE(cache.YakuBit("1,1,1,1,1,1,1,1,1,1,1,1,2") & kBitCompletedThirteenOrphans);
    EXPECT_FALSE(cache.YakuBit("2,111,111,111,111") & kBitCompletedThirteenOrphans);
    // big_four_winds, // 大四喜
    EXPECT_TRUE(cache.YakuBit("2,3,3,3,3") & kBitBigFourWinds);
    EXPECT_FALSE(cache.YakuBit("2,3,3,33") & kBitBigFourWinds);
    // little_four_winds, // 小四喜
    EXPECT_TRUE(cache.YakuBit("2,3,3,3,111") & kBitLittleFourWinds);
    EXPECT_FALSE(cache.YakuBit("2,3,33,111") & kBitLittleFourWinds);
    // four_kans, // 四槓子
    EXPECT_TRUE(cache.YakuBit("2,3,3,3,3") & kBitFourKans);
    EXPECT_FALSE(cache.YakuBit("2,11433") & kBitFourKans);
    EXPECT_FALSE(cache.YakuBit("2,3,3,3,111") & kBitFourKans);
    // dora, // ドラ
    // reversed_dora, // 裏ドラ
    // red_dora, // 赤ドラ

    // Stats for each yaku
    cache.ShowStats(kBitFullyConcealedHand, "門前清自摸和");
    cache.ShowStats(kBitRiichi, "立直");
    cache.ShowStats(kBitIppatsu, "一発");
    cache.ShowStats(kBitRobbingKan, "槍槓");
    cache.ShowStats(kBitAfterKan, "嶺上開花");
    cache.ShowStats(kBitBottomOfTheSea, "海底摸月");
    cache.ShowStats(kBitBottomOfTheRiver, "河底撈魚");
    cache.ShowStats(kBitPinfu, "平和");
    cache.ShowStats(kBitAllSimples, "断幺九");
    cache.ShowStats(kBitPureDoubleChis, "一盃口");
    cache.ShowStats(kBitSeatWindEast, "自風 東");
    cache.ShowStats(kBitSeatWindSouth, "自風 南");
    cache.ShowStats(kBitSeatWindWest, "自風 西");
    cache.ShowStats(kBitSeatWindNorth, "自風 北");
    cache.ShowStats(kBitPrevalentWindEast, "場風 東");
    cache.ShowStats(kBitPrevalentWindSouth, "場風 南");
    cache.ShowStats(kBitPrevalentWindWest, "場風 西");
    cache.ShowStats(kBitPrevalentWindNorth, "場風 北");
    cache.ShowStats(kBitWhiteDragon, "役牌 白");
    cache.ShowStats(kBitGreenDragon, "役牌 發");
    cache.ShowStats(kBitRedDragon, "役牌 中");
    cache.ShowStats(kBitDoubleRiichi, "両立直");
    cache.ShowStats(kBitSevenPairs, "七対子");
    cache.ShowStats(kBitOutsideHand, "混全帯幺九");
    cache.ShowStats(kBitPureStraight, "一気通貫");
    cache.ShowStats(kBitMixedTripleChis, "三色同順");
    cache.ShowStats(kBitTriplePons, "三色同刻");
    cache.ShowStats(kBitThreeKans, "三槓子");
    cache.ShowStats(kBitAllPons, "対々和");
    cache.ShowStats(kBitThreeConcealedPons, "三暗刻");
    cache.ShowStats(kBitLittleThreeDragons, "小三元");
    cache.ShowStats(kBitAllTermsAndHonours, "混老頭");
    cache.ShowStats(kBitTwicePureDoubleChis, "二盃口");
    cache.ShowStats(kBitTerminalsInAllSets, "純全帯幺九");
    cache.ShowStats(kBitHalfFlush, "混一色");
    cache.ShowStats(kBitFullFlush, "清一色");
    cache.ShowStats(kBitBlessingOfMan, "人和");
    cache.ShowStats(kBitBlessingOfHeaven, "天和");
    cache.ShowStats(kBitBlessingOfEarth, "地和");
    cache.ShowStats(kBitBigThreeDragons, "大三元");
    cache.ShowStats(kBitFourConcealedPons, "四暗刻");
    cache.ShowStats(kBitCompletedFourConcealedPons, "四暗刻単騎");
    cache.ShowStats(kBitAllHonours, "字一色");
    cache.ShowStats(kBitAllGreen, "緑一色");
    cache.ShowStats(kBitAllTerminals, "清老頭");
    cache.ShowStats(kBitNineGates, "九蓮宝燈");
    cache.ShowStats(kBitPureNineGates, "純正九蓮宝燈");
    cache.ShowStats(kBitThirteenOrphans, "国士無双");
    cache.ShowStats(kBitCompletedThirteenOrphans, "国士無双１３面");
    cache.ShowStats(kBitBigFourWinds, "大四喜");
    cache.ShowStats(kBitLittleFourWinds, "小四喜");
    cache.ShowStats(kBitFourKans, "四槓子");
    cache.ShowStats(kBitDora, "ドラ");
    cache.ShowStats(kBitReversedDora, "裏ドラ");
    cache.ShowStats(kBitRedDora, "赤ドラ");
}
