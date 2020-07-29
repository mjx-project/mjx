#ifndef MAHJONG_TYPES_H
#define MAHJONG_TYPES_H

#include <cstdint>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

#include <mahjong.pb.h>

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
        kGreen,
        kEmpty
    };

    enum class EventType : std::uint8_t {
        kDraw = mjproto::EVENT_TYPE_DRAW,
        kDiscardFromHand = mjproto::EVENT_TYPE_DISCARD_FROM_HAND,
        kDiscardDrawnTile = mjproto::EVENT_TYPE_DISCARD_DRAWN_TILE,  // ツモ切り, Tsumogiri
        kRiichi = mjproto::EVENT_TYPE_RIICHI,
        kTsumo = mjproto::EVENT_TYPE_TSUMO,
        kRon = mjproto::EVENT_TYPE_RON,
        kChi = mjproto::EVENT_TYPE_CHI,
        kPon = mjproto::EVENT_TYPE_PON,
        kKanClosed = mjproto::EVENT_TYPE_KAN_CLOSED,
        kKanOpened = mjproto::EVENT_TYPE_KAN_OPENED,
        kKanAdded = mjproto::EVENT_TYPE_KAN_ADDED,
        kNewDora = mjproto::EVENT_TYPE_NEW_DORA,
        kRiichiScoreChange = mjproto::EVENT_TYPE_RIICHI_SCORE_CHANGE,
    };

    enum class ActionType : std::uint8_t {
        kDiscard = mjproto::ACTION_TYPE_DISCARD,
        kRiichi = mjproto::ACTION_TYPE_RIICHI,
        kTsumo = mjproto::ACTION_TYPE_TSUMO,
        kRon = mjproto::ACTION_TYPE_RON,
        kChi = mjproto::ACTION_TYPE_CHI,
        kPon = mjproto::ACTION_TYPE_PON,
        kKanClosed = mjproto::ACTION_TYPE_KAN_CLOSED,
        kKanOpened = mjproto::ACTION_TYPE_KAN_OPENED,
        kKanAdded = mjproto::ACTION_TYPE_KAN_ADDED,
        kKyushu = mjproto::ACTION_TYPE_KYUSYU,
    };

    enum class HandStage : std::uint8_t {
        kAfterDiscards,
        kAfterDraw,
        kAfterDrawAfterKan,
        kAfterRiichi,
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

    enum class RoundStage: std::uint8_t {
        kAfterDiscards,
        kAfterDraw,
        kAfterDrawAfterKan,
        kAfterRiichiAndDiscard,
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
        kInitEast = mjproto::ABSOLUTE_POS_INIT_EAST,  // 起家
        kInitSouth = mjproto::ABSOLUTE_POS_INIT_SOUTH,
        kInitWest = mjproto::ABSOLUTE_POS_INIT_WEST,
        kInitNorth = mjproto::ABSOLUTE_POS_INIT_NORTH,  // ラス親
        kEnd,  // Dummy
        kBegin = 0
    };

    enum class RelativePos : std::uint8_t  // Order follows mjlog
    {
        kSelf = mjproto::RELATIVE_POS_SELF,
        kRight = mjproto::RELATIVE_POS_RIGHT,  // 下家
        kMid = mjproto::RELATIVE_POS_MID,      // 対面
        kLeft = mjproto::RELATIVE_POS_LEFT     // 上家
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
        kEast = mjproto::WIND_EAST,
        kSouth = mjproto::WIND_SOUTH,
        kWest = mjproto::WIND_WEST,
        kNorth = mjproto::WIND_NORTH,
    };

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
        kEnd,  // Dummy
        kBegin = 0,
    };

    std::uint8_t Num(TileType type) noexcept ;
    bool Is(TileType type, TileSetType tile_set_type) noexcept;
    TileSetType Color(TileType type) noexcept ;

    using TileTypeCount = std::map<TileType, int>;

}  // namespace mj

#endif //MAHJONG_TYPES_H
