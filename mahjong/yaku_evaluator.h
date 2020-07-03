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
        [[nodiscard]] std::map<Yaku,int> MaximizeTotalFan(const Hand& hand) const noexcept ;

        [[nodiscard]] static bool HasBigThreeDragons(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasAllHonours(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasAllGreen(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasAllTerminals(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasBigFourWinds(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasLittleFourWinds(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasThirteenOrphans(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasCompletedThirteenOrphans(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasNineGates(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasPureNineGates(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasFourKans(const Hand& hand) noexcept;
        [[nodiscard]] static bool HasFourConcealedPons(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static bool HasCompletedFourConcealedPons(const Hand& hand, const TileTypeCount& count) noexcept ;

        [[nodiscard]] static std::optional<int> HasFullyConcealdHand(const Hand& hand) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllSimples(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasWhiteDragon(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasGreenDragon(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasRedDragon(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllTermsAndHonours(const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasHalfFlush(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasFullFlush(const Hand& hand, const TileTypeCount& count) noexcept ;
        [[nodiscard]] static std::optional<int> HasThreeKans(const Hand& hand) noexcept ;
        [[nodiscard]] static std::optional<int> HasPinfu(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasPureDoubleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasTwicePureDoubleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasSevenPairs(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllPons(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasPureStraight(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasMixedTripleChis(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasTriplePons(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasOutsideHand(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasTerminalsInAllSets(
                const Hand &hand,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        WinningHandCache win_cache_;
    };
} // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
