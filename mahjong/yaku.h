#ifndef MAHJONG_YAKU_H
#define MAHJONG_YAKU_H

#include <cstdint>
#include <array>

#include <consts.h>
#include <hand.h>

namespace mj
{
    class Yaku
    {
    public:
        Yaku() = default;
        Yaku(Hand hand, relative_pos from);
        std::unique_ptr<std::vector<std::pair<yaku, std::uint8_t>>> to_vector();  // follows tenhou format
    private:
        std::uint64_t bits_ = 0;
        std::uint8_t dora_count_ = -1;
        std::uint8_t reversed_dora_count_ = -1;
        std::uint8_t red_dora_count_ = -1;
        bool is_valid();
    };
    // fully_concealed_hand, // 門前清自摸和
    // riichi, // 立直
    // ippatsu, // 一発
    // robbing_a_kong, // 槍槓
    // after_a_kong, // 嶺上開花
    // bottom_of_the_sea, // 海底摸月
    // bottom_of_the_river, // 河底撈魚
    // pinfu, // 平和
    // all_simples, // 断幺九
    // pure_double_chows, // 一盃口
    // seat_wind_east,// 自風 東
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
    // outside_hand, // 混全帯幺九
    // pure_straight, // 一気通貫
    // mixed_triple_chows, // 三色同順
    // triple_pungs, // 三色同刻
    // three_kongs, // 三槓子
    // all_pungs, // 対々和
    // three_concealed_pungs, // 三暗刻
    // little_three_dragons, // 小三元
    // all_terms_and_honours, // 混老頭
    // twice_pure_double_chows, // 二盃口
    // terminals_in_all_sets, // 純全帯幺九
    // half_flush, // 混一色
    // full_flush, // 清一色
    // blessing_of_man, // 人和
    // blessing_of_heaven, // 天和
    // blessing_of_earth, // 地和
    // big_three_dragons, // 大三元
    // four_concealed_pungs, // 四暗刻
    // completed_four_concealed_pungs, // 四暗刻単騎
    // all_honours, // 字一色
    // all_green, // 緑一色
    // all_terminals, // 清老頭
    // nine_gates, // 九蓮宝燈
    // pure_nine_gates, // 純正九蓮宝燈
    // thirteen_orphans, // 国士無双
    // completed_thirteen_orphans, // 国士無双１３面
    // big_four_winds, // 大四喜
    // little_four_winds, // 小四喜
    // four_kongs, // 四槓子
    // dora, // ドラ
    // reversed_dora, // 裏ドラ
    // red_dora, // 赤ドラ
}  // namespace mj

#endif //MAHJONG_YAKU_H
