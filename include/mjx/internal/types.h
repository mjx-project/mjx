#ifndef MAHJONG_TYPES_H
#define MAHJONG_TYPES_H

#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "mjx/internal/mjx.pb.h"

namespace mjx::internal {
using TileId =
    std::uint8_t;  // {0, ..., 135} corresponds to mjlog format of Tenhou
// m1   0,   1,   2,   3
// m2   4,   5,   6,   7
// m3   8,   9,  10,  11
// m4  12,  13,  14,  15
// m5  16,  17,  18,  19
// m6  20,  21,  22,  23
// m7  24,  25,  26,  27
// m8  28,  29,  30,  31
// m9  32,  33,  34,  35
// p1  36,  37,  38,  39
// p2  40,  41,  42,  43
// p3  44,  45,  46,  47
// p4  48,  49,  50,  51
// p5  52,  53,  54,  55
// p6  56,  57,  58,  59
// p7  60,  61,  62,  63
// p8  64,  65,  66,  67
// p9  68,  69,  70,  71
// s1  72,  73,  74,  75
// s2  76,  77,  78,  79
// s3  80,  81,  82,  83
// s4  84,  85,  86,  87
// s5  88,  89,  90,  91
// s6  92,  93,  94,  95
// s7  96,  97,  98,  99
// s8 100, 101, 102, 103
// s9 104, 105, 106, 107
// ew 108, 109, 110, 111
// sw 112, 113, 114, 115
// ww 116, 117, 118, 119
// nw 120, 121, 122, 123
// wd 124, 125, 126, 127
// gd 128, 129, 130, 131
// rd 132, 133, 134, 135

// Naming follows
// https://github.com/NegativeMjark/tenhou-log/blob/master/TenhouDecoder.py
enum class TileType : std::uint8_t {
  kM1,
  kM2,
  kM3,
  kM4,
  kM5,
  kM6,
  kM7,
  kM8,
  kM9,
  kP1,
  kP2,
  kP3,
  kP4,
  kP5,
  kP6,
  kP7,
  kP8,
  kP9,
  kS1,
  kS2,
  kS3,
  kS4,
  kS5,
  kS6,
  kS7,
  kS8,
  kS9,
  kEW,
  kSW,
  kWW,
  kNW,
  kWD,
  kGD,
  kRD  // {0, ..., 33}
};

using TileTypeCount = std::map<TileType, int>;

enum class TileSetType : std::uint8_t {  // TODO: rename using honours and
                                         // terminals,
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
  kGreen,
  kEmpty
};

enum class HandStage : std::uint8_t {
  kAfterDiscards,
  kAfterDraw,
  kAfterDrawAfterKan,
  kAfterRiichi,
  kAfterTsumo,
  kAfterTsumoAfterKan,
  kAfterRon,
  kAfterChi,
  kAfterPon,
  kAfterKanOpened,
  kAfterKanClosed,
  kAfterKanAdded,
};

enum class RoundStage : std::uint8_t {
  kAfterDiscard,
  kAfterDraw,
  kAfterDrawAfterKan,
  kAfterRiichi,
  kAfterDiscardAfterRiichi,
  kAfterTsumo,
  kAfterTsumoAfterKan,
  kAfterRon,
  kAfterRonAfterOthersKan,
  kAfterChi,
  kAfterPon,
  kAfterKanOpened,
  kAfterKanClosed,
  kAfterKanAdded,
};

enum class AbsolutePos : std::uint8_t {
  kInitEast = 0,  // 起家
  kInitSouth = 1,
  kInitWest = 2,
  kInitNorth = 3,  // ラス親
  kEnd,            // Dummy
  kBegin = 0
};

enum class RelativePos : std::uint8_t  // Order follows mjlog
{
  kSelf = 0,
  kRight = 1,  // 下家
  kMid = 2,    // 対面
  kLeft = 3    // 上家
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

enum class Wind : std::uint8_t {
  kEast = 0,
  kSouth = 1,
  kWest = 2,
  kNorth = 3,
};

// The yaku No. follows: http://tenhou.net/1/script/tenhou.js
// The terminology basically follows:
// http://mahjong-europe.org/portal/images/docs/riichi_scoresheet_EN.pdf
enum class Yaku : std::uint8_t {
  // 1fan
  kFullyConcealedHand,  //  0. 門前清自摸和
  kRiichi,              //  1. 立直
  kIppatsu,             //  2. 一発
  kRobbingKan,          //  3. 槍槓
  kAfterKan,            //  4. 嶺上開花
  kBottomOfTheSea,      //  5. 海底摸月
  kBottomOfTheRiver,    //  6. 河底撈魚
  kPinfu,               //  7. 平和
  kAllSimples,          //  8. 断幺九
  kPureDoubleChis,      //  9. 一盃口
  kSeatWindEast,        // 10. 自風 東
  kSeatWindSouth,       // 11. 自風 南
  kSeatWindWest,        // 12. 自風 西
  kSeatWindNorth,       // 13. 自風 北
  kPrevalentWindEast,   // 14. 場風 東
  kPrevalentWindSouth,  // 15. 場風 南
  kPrevalentWindWest,   // 16. 場風 西
  kPrevalentWindNorth,  // 17. 場風 北
  kWhiteDragon,         // 18. 役牌 白
  kGreenDragon,         // 19. 役牌 發
  kRedDragon,           // 20. 役牌 中
  // 2 fan
  kDoubleRiichi,        // 21. 両立直
  kSevenPairs,          // 22. 七対子
  kOutsideHand,         // 23. 混全帯幺九
  kPureStraight,        // 24. 一気通貫
  kMixedTripleChis,     // 25. 三色同順
  kTriplePons,          // 26. 三色同刻
  kThreeKans,           // 27. 三槓子
  kAllPons,             // 28. 対々和
  kThreeConcealedPons,  // 29. 三暗刻
  kLittleThreeDragons,  // 30. 小三元
  kAllTermsAndHonours,  // 31. 混老頭
  // 3 fan
  kTwicePureDoubleChis,  // 32. 二盃口
  kTerminalsInAllSets,   // 33. 純全帯幺九
  kHalfFlush,            // 34. 混一色
  // 6 fan
  kFullFlush,  // 35. 清一色
  // mangan
  kBlessingOfMan,  // 36. 人和
  // yakuman
  kBlessingOfHeaven,            // 37. 天和
  kBlessingOfEarth,             // 38. 地和
  kBigThreeDragons,             // 39. 大三元
  kFourConcealedPons,           // 40. 四暗刻
  kCompletedFourConcealedPons,  // 41. 四暗刻単騎
  kAllHonours,                  // 42. 字一色
  kAllGreen,                    // 43. 緑一色
  kAllTerminals,                // 44. 清老頭
  kNineGates,                   // 45. 九蓮宝燈
  kPureNineGates,               // 46. 純正九蓮宝燈
  kThirteenOrphans,             // 47. 国士無双
  kCompletedThirteenOrphans,    // 48. 国士無双１３面
  kBigFourWinds,                // 49. 大四喜
  kLittleFourWinds,             // 50. 小四喜
  kFourKans,                    // 51. 四槓子
  // dora
  kDora,          // 52. ドラ
  kReversedDora,  // 53. 裏ドラ
  kRedDora,       // 54. 赤ドラ
  kEnd,           // Dummy
  kBegin = 0,
};

using PlayerId = std::string;

std::uint8_t Num(TileType type) noexcept;
bool Is(TileType type, TileSetType tile_set_type) noexcept;
TileSetType Color(TileType type) noexcept;

RelativePos ToRelativePos(AbsolutePos origin, AbsolutePos target);
Wind ToSeatWind(AbsolutePos who, AbsolutePos dealer);
mjxproto::EventType OpenTypeToEventType(OpenType open_type);
mjxproto::ActionType OpenTypeToActionType(OpenType open_type);
bool IsSameWind(TileType tile_type, Wind wind);
TileType IndicatorToDora(TileType tile_type);
}  // namespace mjx::internal

#endif  // MAHJONG_TYPES_H
