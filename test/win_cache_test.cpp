#include <numeric>
#include "gtest/gtest.h"
#include "consts.h"
#include "win_cache.h"

using namespace mj;


TEST(win_cache, hand_made_samples)
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
    EXPECT_TRUE(cache.Yaku("2,111,111111111") & bit_pinfu);
    EXPECT_FALSE(cache.Yaku("2,3,111111111") & bit_pinfu);
    // all_simples, // 断幺九
    EXPECT_TRUE(cache.Yaku("2,111,111,111111") & bit_all_simples);
    EXPECT_FALSE(cache.Yaku("2,111,111111111") & bit_all_simples);
    // pure_double_chis, // 一盃口
    EXPECT_TRUE(cache.Yaku("2,222222") & bit_all_simples);
    EXPECT_FALSE(cache.Yaku("2,2,22,222") & bit_all_simples);
    // seat_wind_east,// 自風 東
    EXPECT_TRUE(cache.Yaku("2,3,111111111") & bit_seat_wind_east);
    EXPECT_FALSE(cache.Yaku("2,111,111111111") & bit_seat_wind_east);
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
    EXPECT_TRUE(cache.Yaku("2222222") & bit_seven_pairs);
    EXPECT_FALSE(cache.Yaku("222422") & bit_seven_pairs);
    // outside_hand, // 混全帯幺九
    EXPECT_TRUE(cache.Yaku("2,3,3,111,111") & bit_outside_hand);
    EXPECT_FALSE(cache.Yaku("2,3,3,111111") & bit_outside_hand);
    // pure_straight, // 一気通貫
    EXPECT_TRUE(cache.Yaku("431111111") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("111,431111") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("111,141,1112") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("1413113") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("1314221") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("3,3,1421") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("3,311,11211") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("31113131") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("111,1221131") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("141112112") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("33,112112") & bit_pure_straight);
    EXPECT_FALSE(cache.Yaku("3,3,111,1112") & bit_pure_straight);
    // // mixed_triple_chis, // 三色同順
    EXPECT_TRUE(cache.Yaku("2,111,111,114") & bit_mixed_triple_chis);
    EXPECT_FALSE(cache.Yaku("2,33,111,111") & bit_seat_wind_east);
    // triple_pons, // 三色同刻
    EXPECT_TRUE(cache.Yaku("2,3,3,114") & bit_triple_pons);
    EXPECT_FALSE(cache.Yaku("2,111,111,114") & bit_triple_pons);
    EXPECT_FALSE(cache.Yaku("2,3,333") & bit_triple_pons);
    // three_kans, // 三槓子
    EXPECT_TRUE(cache.Yaku("2,3,3,1113") & bit_three_kans);
    EXPECT_FALSE(cache.Yaku("2,3,3,114") & bit_three_kans);
    // all_pons, // 対々和
    EXPECT_TRUE(cache.Yaku("2,3,333") & bit_all_pons);
    EXPECT_FALSE(cache.Yaku("2,111,333") & bit_all_pons);
    // three_concealed_pons, // 三暗刻
    EXPECT_TRUE(cache.Yaku("2,111,333") & bit_three_concealed_pons);
    EXPECT_FALSE(cache.Yaku("2,111,11133") & bit_three_concealed_pons);
    // little_three_dragons, // 小三元
    EXPECT_TRUE(cache.Yaku("2,3,3,3,111") & bit_little_three_dragons);
    EXPECT_FALSE(cache.Yaku("2,111,333") & bit_little_three_dragons);
    // all_terms_and_honours, // 混老頭
    // EXPECT_TRUE(cache.yaku("2,3,3,3,3") & bit_all_terms_and_honours);
    EXPECT_FALSE(cache.Yaku("2,3,333") & bit_all_terms_and_honours);
    // twice_pure_double_chis, // 二盃口
    EXPECT_TRUE(cache.Yaku("2,222,222") & bit_twice_pure_double_chis);
    EXPECT_FALSE(cache.Yaku("2,22,2222") & bit_twice_pure_double_chis);
    // terminals_in_all_sets, // 純全帯幺九
    EXPECT_TRUE(cache.Yaku("2,111,111,111,111") & bit_terminals_in_all_sets);
    EXPECT_FALSE(cache.Yaku("2,111,111,111111") & bit_terminals_in_all_sets);
    EXPECT_FALSE(cache.Yaku("3,3,4112") & bit_terminals_in_all_sets);
    EXPECT_FALSE(cache.Yaku("111,2333") & bit_terminals_in_all_sets);
    EXPECT_FALSE(cache.Yaku("111,111,2114") & bit_terminals_in_all_sets);
    EXPECT_FALSE(cache.Yaku("3,3,314") & bit_terminals_in_all_sets);
    // half_flush, // 混一色
    EXPECT_TRUE(cache.Yaku("2,3,111111111") & bit_half_flush);
    EXPECT_FALSE(cache.Yaku("23,111111111") & bit_half_flush);
    // full_flush, // 清一色
    EXPECT_TRUE(cache.Yaku("222111113") & bit_full_flush);
    EXPECT_FALSE(cache.Yaku("222,111113") & bit_full_flush);
    // blessing_of_man, // 人和
    // blessing_of_heaven, // 天和
    // blessing_of_earth, // 地和
    // big_three_dragons, // 大三元
    EXPECT_TRUE(cache.Yaku("3,3,3,23") & bit_big_three_dragons);
    EXPECT_FALSE(cache.Yaku("3,3,233") & bit_big_three_dragons);
    // // four_concealed_pons, // 四暗刻
    EXPECT_TRUE(cache.Yaku("23333") & bit_four_concealed_pons);
    EXPECT_FALSE(cache.Yaku("111,2333") & bit_four_concealed_pons);
    // completed_four_concealed_pons, // 四暗刻単騎
    EXPECT_TRUE(cache.Yaku("23333") & bit_completed_four_concealed_pons);
    EXPECT_FALSE(cache.Yaku("111,2333") & bit_completed_four_concealed_pons);
    // all_honours, // 字一色
    EXPECT_TRUE(cache.Yaku("2,3,3,3,3") & bit_all_honours);
    EXPECT_FALSE(cache.Yaku("3,3,3,23") & bit_all_honours);
    // all_green, // 緑一色
    EXPECT_TRUE(cache.Yaku("2,3,3,222") & bit_all_green);
    EXPECT_FALSE(cache.Yaku("2,33,222") & bit_all_green);
    // all_terminals, // 清老頭
    EXPECT_TRUE(cache.Yaku("2,3,3,3,3") & bit_all_terminals);
    EXPECT_FALSE(cache.Yaku("2,3,3,33") & bit_all_terminals);
    // nine_gates, // 九蓮宝燈
    EXPECT_TRUE(cache.Yaku("321111113") & bit_nine_gates);
    EXPECT_FALSE(cache.Yaku("222111113") & bit_nine_gates);
    // pure_nine_gates, // 純正九蓮宝燈
    EXPECT_TRUE(cache.Yaku("321111113") & bit_pure_nine_gates);
    EXPECT_FALSE(cache.Yaku("222111113") & bit_pure_nine_gates);
    // thirteen_orphans, // 国士無双
    EXPECT_TRUE(cache.Yaku("1,1,1,1,1,1,1,1,1,1,1,1,2") & bit_thirteen_orphans);
    EXPECT_FALSE(cache.Yaku("2,111,111,111,111") & bit_thirteen_orphans);
    // completed_thirteen_orphans, // 国士無双１３面
    EXPECT_TRUE(cache.Yaku("1,1,1,1,1,1,1,1,1,1,1,1,2") & bit_completed_thirteen_orphans);
    EXPECT_FALSE(cache.Yaku("2,111,111,111,111") & bit_completed_thirteen_orphans);
    // big_four_winds, // 大四喜
    EXPECT_TRUE(cache.Yaku("2,3,3,3,3") & bit_big_four_winds);
    EXPECT_FALSE(cache.Yaku("2,3,3,33") & bit_big_four_winds);
    // little_four_winds, // 小四喜
    EXPECT_TRUE(cache.Yaku("2,3,3,3,111") & bit_little_four_winds);
    EXPECT_FALSE(cache.Yaku("2,3,33,111") & bit_little_four_winds);
    // four_kans, // 四槓子
    EXPECT_TRUE(cache.Yaku("2,3,3,3,3") & bit_four_kans);
    EXPECT_FALSE(cache.Yaku("2,11433") & bit_four_kans);
    EXPECT_FALSE(cache.Yaku("2,3,3,3,111") & bit_four_kans);
    // dora, // ドラ
    // reversed_dora, // 裏ドラ
    // red_dora, // 赤ドラ

    // Stats for each yaku
    cache.ShowStats(bit_fully_concealed_hand, "門前清自摸和");
    cache.ShowStats(bit_riichi, "立直");
    cache.ShowStats(bit_ippatsu, "一発");
    cache.ShowStats(bit_robbing_a_kan, "槍槓");
    cache.ShowStats(bit_after_a_kan, "嶺上開花");
    cache.ShowStats(bit_bottom_of_the_sea, "海底摸月");
    cache.ShowStats(bit_bottom_of_the_river, "河底撈魚");
    cache.ShowStats(bit_pinfu, "平和");
    cache.ShowStats(bit_all_simples, "断幺九");
    cache.ShowStats(bit_pure_double_chis, "一盃口");
    cache.ShowStats(bit_seat_wind_east, "自風 東");
    cache.ShowStats(bit_seat_wind_south, "自風 南");
    cache.ShowStats(bit_seat_wind_west, "自風 西");
    cache.ShowStats(bit_seat_wind_north, "自風 北");
    cache.ShowStats(bit_prevalent_wind_east, "場風 東");
    cache.ShowStats(bit_prevalent_wind_south, "場風 南");
    cache.ShowStats(bit_prevalent_wind_west, "場風 西");
    cache.ShowStats(bit_prevalent_wind_north, "場風 北");
    cache.ShowStats(bit_white_dragon, "役牌 白");
    cache.ShowStats(bit_green_dragon, "役牌 發");
    cache.ShowStats(bit_red_dragon, "役牌 中");
    cache.ShowStats(bit_double_riichi, "両立直");
    cache.ShowStats(bit_seven_pairs, "七対子");
    cache.ShowStats(bit_outside_hand, "混全帯幺九");
    cache.ShowStats(bit_pure_straight, "一気通貫");
    cache.ShowStats(bit_mixed_triple_chis, "三色同順");
    cache.ShowStats(bit_triple_pons, "三色同刻");
    cache.ShowStats(bit_three_kans, "三槓子");
    cache.ShowStats(bit_all_pons, "対々和");
    cache.ShowStats(bit_three_concealed_pons, "三暗刻");
    cache.ShowStats(bit_little_three_dragons, "小三元");
    cache.ShowStats(bit_all_terms_and_honours, "混老頭");
    cache.ShowStats(bit_twice_pure_double_chis, "二盃口");
    cache.ShowStats(bit_terminals_in_all_sets, "純全帯幺九");
    cache.ShowStats(bit_half_flush, "混一色");
    cache.ShowStats(bit_full_flush, "清一色");
    cache.ShowStats(bit_blessing_of_man, "人和");
    cache.ShowStats(bit_blessing_of_heaven, "天和");
    cache.ShowStats(bit_blessing_of_earth, "地和");
    cache.ShowStats(bit_big_three_dragons, "大三元");
    cache.ShowStats(bit_four_concealed_pons, "四暗刻");
    cache.ShowStats(bit_completed_four_concealed_pons, "四暗刻単騎");
    cache.ShowStats(bit_all_honours, "字一色");
    cache.ShowStats(bit_all_green, "緑一色");
    cache.ShowStats(bit_all_terminals, "清老頭");
    cache.ShowStats(bit_nine_gates, "九蓮宝燈");
    cache.ShowStats(bit_pure_nine_gates, "純正九蓮宝燈");
    cache.ShowStats(bit_thirteen_orphans, "国士無双");
    cache.ShowStats(bit_completed_thirteen_orphans, "国士無双１３面");
    cache.ShowStats(bit_big_four_winds, "大四喜");
    cache.ShowStats(bit_little_four_winds, "小四喜");
    cache.ShowStats(bit_four_kans, "四槓子");
    cache.ShowStats(bit_dora, "ドラ");
    cache.ShowStats(bit_reversed_dora, "裏ドラ");
    cache.ShowStats(bit_red_dora, "赤ドラ");
}
