#ifndef MAHJONG_YAKU_EVALUATOR_H
#define MAHJONG_YAKU_EVALUATOR_H

#include <vector>

#include "types.h"
#include "hand.h"
#include "win_cache.h"
#include "win_score.h"

namespace mj
{
    class YakuEvaluator {
    public:
        YakuEvaluator();
        [[nodiscard]] bool Has(const Hand& hand) const noexcept ;
        [[nodiscard]] WinningScore Eval(const Hand& hand) const noexcept ;

    private:
        static TileTypeCount ClosedHandTiles(const Hand& hand) noexcept ;
        static TileTypeCount ClosedAndOpenedHandTiles(const Hand& hand) noexcept ;
        static int TotalFan(const std::map<Yaku,int>& yaku) noexcept ;
        std::map<Yaku,int> MaximizeTotalFan(const Hand& hand) const noexcept ;

        [[nodiscard]] bool HasBigThreeDragons(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] bool HasAllHonours(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] bool HasAllGreen(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] bool HasAllTerminals(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] bool HasBigFourWinds(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] bool HasLittleFourWinds(const TileTypeCount& count) const noexcept ;

        [[nodiscard]] std::optional<int> HasFullyConcealdHand(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasAllSimples(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasWhiteDragon(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasGreenDragon(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasRedDragon(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasAllTermsAndHonours(const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasHalfFlush(const Hand& hand, const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasFullFlush(const Hand& hand, const TileTypeCount& count) const noexcept ;
        [[nodiscard]] std::optional<int> HasThreeKans(const Hand& hand) const noexcept ;
        [[nodiscard]] std::optional<int> HasPinfu(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasPureDoubleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasTwicePureDoubleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasSevenPairs(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasAllPons(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasPureStraight(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasMixedTripleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasTriplePons(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasOutsideHand(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        [[nodiscard]] std::optional<int> HasTerminalsInAllSets(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) const noexcept ;
        WinningHandCache win_cache_;
    };
} // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
