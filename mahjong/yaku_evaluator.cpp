#include <iostream>
#include "consts.h"
#include "block.h"
#include "yaku_evaluator.h"

namespace mj
{

    YakuEvaluator::YakuEvaluator(const WinningHandCache &win_cache) : win_cache_(win_cache){ }

    std::vector<Yaku> YakuEvaluator::Apply(Hand &hand) {
        auto blocks = Block::Build(hand.ToArray());
        auto blocks_str = Block::BlocksToString(blocks);
        if (!win_cache_.Has(blocks_str)) return std::vector<Yaku>();

        std::vector<Yaku> yakus;
        std::uint64_t yaku_bit = win_cache_.YakuBit(blocks_str);
        // kBitFullyConcealedHand         門前清自摸和
        if (hand.IsMenzen() && hand.LastActionType() == ActionType::kTsumo) yaku_bit |= kBitFullyConcealedHand;
        else yaku_bit &= ~kBitFullyConcealedHand;
        // kBitRiichi                     立直
        // kBitIppatsu                    一発
        // kBitRobbingKan                 槍槓
        // kBitAfterKan                   嶺上開花
        // kBitBottomOfTheSea             海底摸月
        // kBitBottomOfTheRiver           河底撈魚
        // kBitPinfu                      平和
        // kBitAllSimples                 断幺九
        // kBitPureDoubleChis             一盃口
        // kBitSeatWindEast               自風 東
        // kBitSeatWindSouth              自風 南
        // kBitSeatWindWest               自風 西
        // kBitSeatWindNorth              自風 北
        // kBitPrevalentWindEast          場風 東
        // kBitPrevalentWindSouth         場風 南
        // kBitPrevalentWindWest          場風 西
        // kBitPrevalentWindNorth         場風 北
        // kBitWhiteDragon                役牌 白
        // kBitGreenDragon                役牌 發
        // kBitRedDragon                  役牌 中
        // kBitDoubleRiichi               両立直
        // kBitSevenPairs                 七対子
        // kBitOutsideHand                混全帯幺九
        // kBitPureStraight               一気通貫
        // kBitMixedTripleChis            三色同順
        // kBitTriplePons                 三色同刻
        // kBitThreeKans                  三槓子
        // kBitAllPons                    対々和
        // kBitThreeConcealedPons         三暗刻
        // kBitLittleThreeDragons         小三元
        // kBitAllTermsAndHonours         混老頭
        // kBitTwicePureDoubleChis        二盃口
        // kBitTerminalsInAllSets         純全帯幺九
        // kBitHalfFlush                  混一色
        // kBitFullFlush                  清一色
        // kBitBlessingOfMan              人和
        // kBitBlessingOfHeaven           天和
        // kBitBlessingOfEarth            地和
        // kBitBigThreeDragons            大三元
        // kBitFourConcealedPons          四暗刻
        // kBitCompletedFourConcealedPons 四暗刻単騎
        // kBitAllHonours                 字一色
        // kBitAllGreen                   緑一色
        // kBitAllTerminals               清老頭
        // kBitNineGates                  九蓮宝燈
        // kBitPureNineGates              純正九蓮宝燈
        // kBitThirteenOrphans            国士無双
        // kBitCompletedThirteenOrphans   国士無双１３面
        // kBitBigFourWinds               大四喜
        // kBitLittleFourWinds            小四喜
        // kBitFourKans                   四槓子
        // kBitDora                       ドラ
        // kBitReversedDora               裏ドラ
        // kBitRedDora                    赤ドラ

        // Push all yaku to vector
        for (std::uint8_t i = 0; Yaku(i) != Yaku::kEnd; ++i) {
            if (yaku_bit&(1<<i)) {
                yakus.emplace_back(Yaku(i));
            }
        }
        return yakus;
    }

    std::uint64_t mj::YakuEvaluator::YakuBit(Hand &hand) {
        return 0;
    }
}