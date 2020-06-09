#ifndef MAHJONG_CONSTS_H
#define MAHJONG_CONSTS_H

#include <cstdint>


namespace mj {
    using TileId = std::uint8_t;  // {0, ..., 135} corresponds to mjlog format of Tenhou

    // Naming follows https://github.com/NegativeMjark/tenhou-log/blob/master/TenhouDecoder.py
    enum class TileType : std::uint8_t {
        kM1, kM2, kM3, kM4, kM5, kM6, kM7, kM8, kM9,
        kP1, kP2, kP3, kP4, kP5, kP6, kP7, kP8, kP9,
        kS1, kS2, kS3, kS4, kS5, kS6, kS7, kS8, kS9,
        kEW, kSW, kWW, kNW,
        kWD, kGD, kRD  // {0, ..., 33}
    };

    enum class TileSetType : std::uint8_t {  // TODO: rename using honours and terminals,
        kAll,
        kManzu,
        kPinzu,
        kSouzu,
        kTanyao,
        kTerminals,
        kWinds,
        kDragons,
        kHonours,
        kYaocyu,
        kRedFive,
        kEmpty
    };

    enum class TilePhase : std::uint8_t {
        kAfterDiscards,
        kAfterDraw,
        kAfterChi,
        kAfterPon,
        kAfterDeclareRiichi,
        kAfterKanOpened,
        kAfterKanClosed,
        kAfterKanAdded,
        kAfterWin
    };

    enum class AbsolutePos : std::uint8_t {
        kEast,
        kSouth,
        kWest,
        kNorth
    };

    enum class RelativePos : std::uint8_t  // Order follows mjlog
    {
        kSelf,
        kRight,  // 下家
        kMid,    // 対面
        kLeft    // 上家
    };

    enum class OpenType : std::uint8_t {
        kChi,
        kPon,
        kKanOpened,  // opened kan（大明槓）
        kKanClosed,  // closed kan（暗槓）
        kKanAdded    // added kan（加槓）
    };

    enum class Fan : std::uint8_t {
        kOne,
        kTwo,
        kThree,
        kFour,
        kMangan,
        kHaneman,
        kBaiman,
        kSanbaiman,
        kYakuman
    };

    using Minipoint = std::uint8_t;

    // The order follows: http://tenhou.net/1/script/tenhou.js
    // The terminology basically follows: http://mahjong-europe.org/portal/images/docs/riichi_scoresheet_EN.pdf
    enum class Yaku : std::uint8_t {
        // 1fan
        kFullyConcealedHand, // 門前清自摸和
        kRiichi, // 立直
        kIppatsu, // 一発
        kRobbingKan, // 槍槓
        kAfterKan, // 嶺上開花
        kBottomOfTheSea, // 海底摸月
        kBottomOfTheRiver, // 河底撈魚
        kPinfu, // 平和
        kAllSimples, // 断幺九
        kPureDoubleChis, // 一盃口
        kSeatWindEast,// 自風 東
        kSeatWindSouth, // 自風 南
        kSeatWindWest, // 自風 西
        kSeatWindNorth, // 自風 北
        kPrevalentWindEast, // 場風 東
        kPrevalentWindSouth, // 場風 南
        kPrevalentWindWest, // 場風 西
        kPrevalentWindNorth, // 場風 北
        kWhiteDragon, // 役牌 白
        kGreenDragon, // 役牌 發
        kRedDragon, // 役牌 中
        // 2 fan
        kDoubleRiichi, // 両立直
        kSevenPairs, // 七対子
        kOutsideHand, // 混全帯幺九
        kPureStraight, // 一気通貫
        kMixedTripleChis, // 三色同順
        kTriplePons, // 三色同刻
        kThreeKans, // 三槓子
        kAllPons, // 対々和
        kThreeConcealedPons, // 三暗刻
        kLittleThreeDragons, // 小三元
        kAllTermsAndHonours, // 混老頭
        // 3 fan
        kTwicePureDoubleChis, // 二盃口
        kTerminalsInAllSets, // 純全帯幺九
        kHalfFlush, // 混一色
        // 6 fan
        kFullFlush, // 清一色
        // mangan
        kBlessingOfMan, // 人和
        // yakuman
        kBlessingOfHeaven, // 天和
        kBlessingOfEarth, // 地和
        kBigThreeDragons, // 大三元
        kFourConcealedPons, // 四暗刻
        kCompletedFourConcealedPons, // 四暗刻単騎
        kAllHonours, // 字一色
        kAllGreen, // 緑一色
        kAllTerminals, // 清老頭
        kNineGates, // 九蓮宝燈
        kPureNineGates, // 純正九蓮宝燈
        kThirteenOrphans, // 国士無双
        kCompletedThirteenOrphans, // 国士無双１３面
        kBigFourWinds, // 大四喜
        kLittleFourWinds, // 小四喜
        kFourKans, // 四槓子
        // dora
        kDora, // ドラ
        kReversedDora, // 裏ドラ
        kRedDora, // 赤ドラ
    };

                                                            //43210987654321098765432109876543210987654321098765432109876543210987654321
    constexpr std::uint64_t kBitFullyConcealedHand=         0b00000000000000000000000000000000000000000000000000000000000000000000000001; // 門前清自摸和
    constexpr std::uint64_t kBitRiichi=                     0b00000000000000000000000000000000000000000000000000000000000000000000000010; // 立直
    constexpr std::uint64_t kBitIppatsu=                    0b00000000000000000000000000000000000000000000000000000000000000000000000100; // 一発
    constexpr std::uint64_t kBitRobbingKan=                 0b00000000000000000000000000000000000000000000000000000000000000000000001000; // 槍槓
    constexpr std::uint64_t kBitAfterKan=                   0b00000000000000000000000000000000000000000000000000000000000000000000010000; // 嶺上開花
    constexpr std::uint64_t kBitBottomOfTheSea=             0b00000000000000000000000000000000000000000000000000000000000000000000100000; // 海底摸月
    constexpr std::uint64_t kBitBottomOfTheRiver=           0b00000000000000000000000000000000000000000000000000000000000000000001000000; // 河底撈魚
    constexpr std::uint64_t kBitPinfu=                      0b00000000000000000000000000000000000000000000000000000000000000000010000000; // 平和
    constexpr std::uint64_t kBitAllSimples=                 0b00000000000000000000000000000000000000000000000000000000000000000100000000; // 断幺九
    constexpr std::uint64_t kBitPureDoubleChis=             0b00000000000000000000000000000000000000000000000000000000000000001000000000; // 一盃口
    constexpr std::uint64_t kBitSeatWindEast=               0b00000000000000000000000000000000000000000000000000000000000000010000000000; // 自風 東
    constexpr std::uint64_t kBitSeatWindSouth=              0b00000000000000000000000000000000000000000000000000000000000000100000000000; // 自風 南
    constexpr std::uint64_t kBitSeatWindWest=               0b00000000000000000000000000000000000000000000000000000000000001000000000000; // 自風 西
    constexpr std::uint64_t kBitSeatWindNorth=              0b00000000000000000000000000000000000000000000000000000000000010000000000000; // 自風 北
    constexpr std::uint64_t kBitPrevalentWindEast=          0b00000000000000000000000000000000000000000000000000000000000100000000000000; // 場風 東
    constexpr std::uint64_t kBitPrevalentWindSouth=         0b00000000000000000000000000000000000000000000000000000000001000000000000000; // 場風 南
    constexpr std::uint64_t kBitPrevalentWindWest=          0b00000000000000000000000000000000000000000000000000000000010000000000000000; // 場風 西
    constexpr std::uint64_t kBitPrevalentWindNorth=         0b00000000000000000000000000000000000000000000000000000000100000000000000000; // 場風 北
    constexpr std::uint64_t kBitWhiteDragon=                0b00000000000000000000000000000000000000000000000000000001000000000000000000; // 役牌 白
    constexpr std::uint64_t kBitGreenDragon=                0b00000000000000000000000000000000000000000000000000000010000000000000000000; // 役牌 發
    constexpr std::uint64_t kBitRedDragon=                  0b00000000000000000000000000000000000000000000000000000100000000000000000000; // 役牌 中
    constexpr std::uint64_t kBitDoubleRiichi=               0b00000000000000000000000000000000000000000000000000001000000000000000000000; // 両立直
    constexpr std::uint64_t kBitSevenPairs=                 0b00000000000000000000000000000000000000000000000000010000000000000000000000; // 七対子
    constexpr std::uint64_t kBitOutsideHand=                0b00000000000000000000000000000000000000000000000000100000000000000000000000; // 混全帯幺九
    constexpr std::uint64_t kBitPureStraight=               0b00000000000000000000000000000000000000000000000001000000000000000000000000; // 一気通貫
    constexpr std::uint64_t kBitMixedTripleChis=            0b00000000000000000000000000000000000000000000000010000000000000000000000000; // 三色同順
    constexpr std::uint64_t kBitTriplePons=                 0b00000000000000000000000000000000000000000000000100000000000000000000000000; // 三色同刻
    constexpr std::uint64_t kBitThreeKans=                  0b00000000000000000000000000000000000000000000001000000000000000000000000000; // 三槓子
    constexpr std::uint64_t kBitAllPons=                    0b00000000000000000000000000000000000000000000010000000000000000000000000000; // 対々和
    constexpr std::uint64_t kBitThreeConcealedPons=         0b00000000000000000000000000000000000000000000100000000000000000000000000000; // 三暗刻
    constexpr std::uint64_t kBitLittleThreeDragons=         0b00000000000000000000000000000000000000000001000000000000000000000000000000; // 小三元
    constexpr std::uint64_t kBitAllTermsAndHonours=         0b00000000000000000000000000000000000000000010000000000000000000000000000000; // 混老頭
    constexpr std::uint64_t kBitTwicePureDoubleChis=        0b00000000000000000000000000000000000000000100000000000000000000000000000000; // 二盃口
    constexpr std::uint64_t kBitTerminalsInAllSets=         0b00000000000000000000000000000000000000001000000000000000000000000000000000; // 純全帯幺九
    constexpr std::uint64_t kBitHalfFlush=                  0b00000000000000000000000000000000000000010000000000000000000000000000000000; // 混一色
    constexpr std::uint64_t kBitFullFlush=                  0b00000000000000000000000000000000000000100000000000000000000000000000000000; // 清一色
    constexpr std::uint64_t kBitBlessingOfMan=              0b00000000000000000000000000000000000001000000000000000000000000000000000000; // 人和
    constexpr std::uint64_t kBitBlessingOfHeaven=           0b00000000000000000000000000000000000010000000000000000000000000000000000000; // 天和
    constexpr std::uint64_t kBitBlessingOfEarth=            0b00000000000000000000000000000000000100000000000000000000000000000000000000; // 地和
    constexpr std::uint64_t kBitBigThreeDragons=            0b00000000000000000000000000000000001000000000000000000000000000000000000000; // 大三元
    constexpr std::uint64_t kBitFourConcealedPons=          0b00000000000000000000000000000000010000000000000000000000000000000000000000; // 四暗刻
    constexpr std::uint64_t kBitCompletedFourConcealedPons= 0b00000000000000000000000000000000100000000000000000000000000000000000000000; // 四暗刻単騎
    constexpr std::uint64_t kBitAllHonours=                 0b00000000000000000000000000000001000000000000000000000000000000000000000000; // 字一色
    constexpr std::uint64_t kBitAllGreen=                   0b00000000000000000000000000000010000000000000000000000000000000000000000000; // 緑一色
    constexpr std::uint64_t kBitAllTerminals=               0b00000000000000000000000000000100000000000000000000000000000000000000000000; // 清老頭
    constexpr std::uint64_t kBitNineGates=                  0b00000000000000000000000000001000000000000000000000000000000000000000000000; // 九蓮宝燈
    constexpr std::uint64_t kBitPureNineGates=              0b00000000000000000000000000010000000000000000000000000000000000000000000000; // 純正九蓮宝燈
    constexpr std::uint64_t kBitThirteenOrphans=            0b00000000000000000000000000100000000000000000000000000000000000000000000000; // 国士無双
    constexpr std::uint64_t kBitCompletedThirteenOrphans=   0b00000000000000000000000001000000000000000000000000000000000000000000000000; // 国士無双１３面
    constexpr std::uint64_t kBitBigFourWinds=               0b00000000000000000000000010000000000000000000000000000000000000000000000000; // 大四喜
    constexpr std::uint64_t kBitLittleFourWinds=            0b00000000000000000000000100000000000000000000000000000000000000000000000000; // 小四喜
    constexpr std::uint64_t kBitFourKans=                   0b00000000000000000000001000000000000000000000000000000000000000000000000000; // 四槓子
    constexpr std::uint64_t kBitDora=                       0b00000000000000000000010000000000000000000000000000000000000000000000000000; // ドラ
    constexpr std::uint64_t kBitReversedDora=               0b00000000000000000000100000000000000000000000000000000000000000000000000000; // 裏ドラ
    constexpr std::uint64_t kBitRedDora=                    0b00000000000000000001000000000000000000000000000000000000000000000000000000; // 赤ドラ
}  // namespace mj

#endif //MAHJONG_CONSTS_H
