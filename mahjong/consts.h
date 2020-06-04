#ifndef MAHJONG_CONSTS_H
#define MAHJONG_CONSTS_H

#include <cstdint>


namespace mj {
    using tile_id = std::uint8_t;  // {0, ..., 135} corresponds to mjlog format of Tenhou

    enum class tile_type
            : std::uint8_t  // Naming follows https://github.com/NegativeMjark/tenhou-log/blob/master/TenhouDecoder.py
    {
        m1, m2, m3, m4, m5, m6, m7, m8, m9,
        p1, p2, p3, p4, p5, p6, p7, p8, p9,
        s1, s2, s3, s4, s5, s6, s7, s8, s9,
        ew, sw, ww, nw,
        wd, gd, rd  // {0, ..., 33}
    };

    enum class tile_set_type : std::uint8_t {  // TODO: rename using honours and terminals,
        all, manzu, pinzu, souzu, tanyao, terminals, winds, dragons, honors, yaochu, red_five, empty
    };

    enum class hand_phase : std::uint8_t {
        after_discard,
        after_draw,
        after_chow,
        after_pung,
        after_declare_riichi,
        after_kong_mld,
        after_kong_cnc,
        after_kong_ext,
        after_win
    };

    enum class ActionType : std::uint8_t {
        discard,
        riichi_and_discard,
        tsumo,
        ron,
        chi,
        pon,
        kan_opened,
        kan_closed,
        kan_added,
        kyushu,  // 9 different honors and terminals （九種九牌）　
    };

    enum class AbsolutePos : std::uint8_t {
        east,
        south,
        west,
        north
    };

    AbsolutePos& operator++(AbsolutePos &e ) {  // ++absolute_pos
        e = static_cast<AbsolutePos>(static_cast<std::underlying_type<AbsolutePos>::type>(e) + 1);
        return e;
    }

    AbsolutePos operator++(AbsolutePos &e, std::int32_t ) {  // absolute_pos++
        AbsolutePos tmp = e;
        ++e;
        return tmp;
    }

    enum class relative_pos : std::uint8_t  // Order follows mjlog
    {
        self,
        right,  // 下家
        mid,    // 対面
        left    // 上家
    };

    enum class open_type : std::uint8_t {
        chow,
        pung,
        kong_mld,  // melded kong （大明槓）
        kong_cnc,  // concealed kong （暗槓）
        kong_ext   // extending kong （加槓）
    };

    enum class fan : std::uint8_t {
        one,
        two,
        three,
        four,
        mangan,
        haneman,
        baiman,
        sanbaiman,
        yakuman
    };

    using minipoint = std::uint8_t;

    // The order follows: http://tenhou.net/1/script/tenhou.js
    // The terminology basically follows: http://mahjong-europe.org/portal/images/docs/riichi_scoresheet_EN.pdf
    enum class yaku : std::uint8_t {
        // 1fan
        fully_concealed_hand, // 門前清自摸和
        riichi, // 立直
        ippatsu, // 一発
        robbing_a_kong, // 槍槓
        after_a_kong, // 嶺上開花
        bottom_of_the_sea, // 海底摸月
        bottom_of_the_river, // 河底撈魚
        pinfu, // 平和
        all_simples, // 断幺九
        pure_double_chows, // 一盃口
        seat_wind_east,// 自風 東
        seat_wind_south, // 自風 南
        seat_wind_west, // 自風 西
        seat_wind_north, // 自風 北
        prevalent_wind_east, // 場風 東
        prevalent_wind_south, // 場風 南
        prevalent_wind_west, // 場風 西
        prevalent_wind_north, // 場風 北
        white_dragon, // 役牌 白
        green_dragon, // 役牌 發
        red_dragon, // 役牌 中
        // 2 fan
        double_riichi, // 両立直
        seven_pairs, // 七対子
        outside_hand, // 混全帯幺九
        pure_straight, // 一気通貫
        mixed_triple_chows, // 三色同順
        triple_pungs, // 三色同刻
        three_kongs, // 三槓子
        all_pungs, // 対々和
        three_concealed_pungs, // 三暗刻
        little_three_dragons, // 小三元
        all_terms_and_honours, // 混老頭
        // 3 fan
        twice_pure_double_chows, // 二盃口
        terminals_in_all_sets, // 純全帯幺九
        half_flush, // 混一色
        // 6 fan
        full_flush, // 清一色
        // mangan
        blessing_of_man, // 人和
        // yakuman
        blessing_of_heaven, // 天和
        blessing_of_earth, // 地和
        big_three_dragons, // 大三元
        four_concealed_pungs, // 四暗刻
        completed_four_concealed_pungs, // 四暗刻単騎
        all_honours, // 字一色
        all_green, // 緑一色
        all_terminals, // 清老頭
        nine_gates, // 九蓮宝燈
        pure_nine_gates, // 純正九蓮宝燈
        thirteen_orphans, // 国士無双
        completed_thirteen_orphans, // 国士無双１３面
        big_four_winds, // 大四喜
        little_four_winds, // 小四喜
        four_kongs, // 四槓子
        dora, // ドラ
        reversed_dora, // 裏ドラ
        red_dora, // 赤ドラ
    };

                                                               //43210987654321098765432109876543210987654321098765432109876543210987654321
    constexpr std::uint64_t bit_fully_concealed_hand=          0b00000000000000000000000000000000000000000000000000000000000000000000000001; // 門前清自摸和
    constexpr std::uint64_t bit_riichi=                        0b00000000000000000000000000000000000000000000000000000000000000000000000010; // riichi, // 立直
    constexpr std::uint64_t bit_ippatsu=                       0b00000000000000000000000000000000000000000000000000000000000000000000000100; // ippatsu, // 一発
    constexpr std::uint64_t bit_robbing_a_kong=                0b00000000000000000000000000000000000000000000000000000000000000000000001000; // robbing_a_kong, // 槍槓
    constexpr std::uint64_t bit_after_a_kong=                  0b00000000000000000000000000000000000000000000000000000000000000000000010000; // after_a_kong, // 嶺上開花
    constexpr std::uint64_t bit_bottom_of_the_sea=             0b00000000000000000000000000000000000000000000000000000000000000000000100000; // bottom_of_the_sea, // 海底摸月
    constexpr std::uint64_t bit_bottom_of_the_river=           0b00000000000000000000000000000000000000000000000000000000000000000001000000; // bottom_of_the_river, // 河底撈魚
    constexpr std::uint64_t bit_pinfu=                         0b00000000000000000000000000000000000000000000000000000000000000000010000000; // pinfu, // 平和
    constexpr std::uint64_t bit_all_simples=                   0b00000000000000000000000000000000000000000000000000000000000000000100000000; // all_simples, // 断幺九
    constexpr std::uint64_t bit_pure_double_chows=             0b00000000000000000000000000000000000000000000000000000000000000001000000000; // pure_double_chows, // 一盃口
    constexpr std::uint64_t bit_seat_wind_east=                0b00000000000000000000000000000000000000000000000000000000000000010000000000; // seat_wind_east,// 自風 東
    constexpr std::uint64_t bit_seat_wind_south=               0b00000000000000000000000000000000000000000000000000000000000000100000000000; // seat_wind_south, // 自風 南
    constexpr std::uint64_t bit_seat_wind_west=                0b00000000000000000000000000000000000000000000000000000000000001000000000000; // seat_wind_west, // 自風 西
    constexpr std::uint64_t bit_seat_wind_north=               0b00000000000000000000000000000000000000000000000000000000000010000000000000; // seat_wind_north, // 自風 北
    constexpr std::uint64_t bit_prevalent_wind_east=           0b00000000000000000000000000000000000000000000000000000000000100000000000000; // prevalent_wind_east, // 場風 東
    constexpr std::uint64_t bit_prevalent_wind_south=          0b00000000000000000000000000000000000000000000000000000000001000000000000000; // prevalent_wind_south, // 場風 南
    constexpr std::uint64_t bit_prevalent_wind_west=           0b00000000000000000000000000000000000000000000000000000000010000000000000000; // prevalent_wind_west, // 場風 西
    constexpr std::uint64_t bit_prevalent_wind_north=          0b00000000000000000000000000000000000000000000000000000000100000000000000000; // prevalent_wind_north, // 場風 北
    constexpr std::uint64_t bit_white_dragon=                  0b00000000000000000000000000000000000000000000000000000001000000000000000000; // white_dragon, // 役牌 白
    constexpr std::uint64_t bit_green_dragon=                  0b00000000000000000000000000000000000000000000000000000010000000000000000000; // green_dragon, // 役牌 發
    constexpr std::uint64_t bit_red_dragon=                    0b00000000000000000000000000000000000000000000000000000100000000000000000000; // red_dragon, // 役牌 中
    constexpr std::uint64_t bit_double_riichi=                 0b00000000000000000000000000000000000000000000000000001000000000000000000000; // double_riichi, // 両立直
    constexpr std::uint64_t bit_seven_pairs=                   0b00000000000000000000000000000000000000000000000000010000000000000000000000; // seven_pairs, // 七対子
    constexpr std::uint64_t bit_outside_hand=                  0b00000000000000000000000000000000000000000000000000100000000000000000000000; // outside_hand, // 混全帯幺九
    constexpr std::uint64_t bit_pure_straight=                 0b00000000000000000000000000000000000000000000000001000000000000000000000000; // pure_straight, // 一気通貫
    constexpr std::uint64_t bit_mixed_triple_chows=            0b00000000000000000000000000000000000000000000000010000000000000000000000000; // mixed_triple_chows, // 三色同順
    constexpr std::uint64_t bit_triple_pungs=                  0b00000000000000000000000000000000000000000000000100000000000000000000000000; // triple_pungs, // 三色同刻
    constexpr std::uint64_t bit_three_kongs=                   0b00000000000000000000000000000000000000000000001000000000000000000000000000; // three_kongs, // 三槓子
    constexpr std::uint64_t bit_all_pungs=                     0b00000000000000000000000000000000000000000000010000000000000000000000000000; // all_pungs, // 対々和
    constexpr std::uint64_t bit_three_concealed_pungs=         0b00000000000000000000000000000000000000000000100000000000000000000000000000; // three_concealed_pungs, // 三暗刻
    constexpr std::uint64_t bit_little_three_dragons=          0b00000000000000000000000000000000000000000001000000000000000000000000000000; // little_three_dragons, // 小三元
    constexpr std::uint64_t bit_all_terms_and_honours=         0b00000000000000000000000000000000000000000010000000000000000000000000000000; // all_terms_and_honours, // 混老頭
    constexpr std::uint64_t bit_twice_pure_double_chows=       0b00000000000000000000000000000000000000000100000000000000000000000000000000; // twice_pure_double_chows, // 二盃口
    constexpr std::uint64_t bit_terminals_in_all_sets=         0b00000000000000000000000000000000000000001000000000000000000000000000000000; // terminals_in_all_sets, // 純全帯幺九
    constexpr std::uint64_t bit_half_flush=                    0b00000000000000000000000000000000000000010000000000000000000000000000000000; // half_flush, // 混一色
    constexpr std::uint64_t bit_full_flush=                    0b00000000000000000000000000000000000000100000000000000000000000000000000000; // full_flush, // 清一色
    constexpr std::uint64_t bit_blessing_of_man=               0b00000000000000000000000000000000000001000000000000000000000000000000000000; // blessing_of_man, // 人和
    constexpr std::uint64_t bit_blessing_of_heaven=            0b00000000000000000000000000000000000010000000000000000000000000000000000000; // blessing_of_heaven, // 天和
    constexpr std::uint64_t bit_blessing_of_earth=             0b00000000000000000000000000000000000100000000000000000000000000000000000000; // blessing_of_earth, // 地和
    constexpr std::uint64_t bit_big_three_dragons=             0b00000000000000000000000000000000001000000000000000000000000000000000000000; // big_three_dragons, // 大三元
    constexpr std::uint64_t bit_four_concealed_pungs=          0b00000000000000000000000000000000010000000000000000000000000000000000000000; // four_concealed_pungs, // 四暗刻
    constexpr std::uint64_t bit_completed_four_concealed_pungs=0b00000000000000000000000000000000100000000000000000000000000000000000000000; // completed_four_concealed_pungs, // 四暗刻単騎
    constexpr std::uint64_t bit_all_honours=                   0b00000000000000000000000000000001000000000000000000000000000000000000000000; // all_honours, // 字一色
    constexpr std::uint64_t bit_all_green=                     0b00000000000000000000000000000010000000000000000000000000000000000000000000; // all_green, // 緑一色
    constexpr std::uint64_t bit_all_terminals=                 0b00000000000000000000000000000100000000000000000000000000000000000000000000; // all_terminals, // 清老頭
    constexpr std::uint64_t bit_nine_gates=                    0b00000000000000000000000000001000000000000000000000000000000000000000000000; // nine_gates, // 九蓮宝燈
    constexpr std::uint64_t bit_pure_nine_gates=               0b00000000000000000000000000010000000000000000000000000000000000000000000000; // pure_nine_gates, // 純正九蓮宝燈
    constexpr std::uint64_t bit_thirteen_orphans=              0b00000000000000000000000000100000000000000000000000000000000000000000000000; // thirteen_orphans, // 国士無双
    constexpr std::uint64_t bit_completed_thirteen_orphans=    0b00000000000000000000000001000000000000000000000000000000000000000000000000; // completed_thirteen_orphans, // 国士無双１３面
    constexpr std::uint64_t bit_big_four_winds=                0b00000000000000000000000010000000000000000000000000000000000000000000000000; // big_four_winds, // 大四喜
    constexpr std::uint64_t bit_little_four_winds=             0b00000000000000000000000100000000000000000000000000000000000000000000000000; // little_four_winds, // 小四喜
    constexpr std::uint64_t bit_four_kongs=                    0b00000000000000000000001000000000000000000000000000000000000000000000000000; // four_kongs, // 四槓子
    constexpr std::uint64_t bit_dora=                          0b00000000000000000000010000000000000000000000000000000000000000000000000000; // dora, // ドラ
    constexpr std::uint64_t bit_reversed_dora=                 0b00000000000000000000100000000000000000000000000000000000000000000000000000; // reversed_dora, // 裏ドラ
    constexpr std::uint64_t bit_red_dora=                      0b00000000000000000001000000000000000000000000000000000000000000000000000000; // red_dora, // 赤ドラ
}  // namespace mj

#endif //MAHJONG_CONSTS_H
