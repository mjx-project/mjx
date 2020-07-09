#ifndef MAHJONG_YAKU_EVALUATOR_H
#define MAHJONG_YAKU_EVALUATOR_H

#include <vector>
#include <tuple>

#include "types.h"
#include "win_info.h"
#include "win_cache.h"
#include "win_score.h"

namespace mj
{
    class YakuEvaluator {
    public:
        YakuEvaluator() = default;
        [[nodiscard]] WinningScore Eval(const WinningInfo& win_info) const noexcept ;
        //[[nodiscard]] bool Has(const Hand& hand) const noexcept ;

    private:
        [[nodiscard]] const WinningHandCache& win_cache() const;

        static void JudgeYakuman(
                const WinningInfo& win_info,
                WinningScore& score) noexcept ;

        static void JudgeSimpleYaku(
                const WinningInfo& win_info,
                WinningScore& score) noexcept ;

        static int TotalFan(const std::map<Yaku,int>& yaku) noexcept ;
        [[nodiscard]] std::tuple<std::map<Yaku,int>,std::vector<TileTypeCount>,std::vector<TileTypeCount>>
        MaximizeTotalFan(const WinningInfo& win_info) const noexcept ;

        [[nodiscard]] static int CalculateFu(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& heads,
                const WinningScore& win_score) noexcept ;

        [[nodiscard]] static bool HasBigThreeDragons(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasAllHonours(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasAllGreen(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasAllTerminals(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasBigFourWinds(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasLittleFourWinds(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasThirteenOrphans(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasCompletedThirteenOrphans(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasNineGates(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasPureNineGates(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasFourKans(const WinningInfo& win_info) noexcept;
        [[nodiscard]] static bool HasFourConcealedPons(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool HasCompletedFourConcealedPons(const WinningInfo& win_info) noexcept ;

        [[nodiscard]] static std::optional<int> HasFullyConcealdHand(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllSimples(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasWhiteDragon(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasGreenDragon(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasRedDragon(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllTermsAndHonours(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasHalfFlush(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasFullFlush(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasThreeKans(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasLittleThreeDragons(const WinningInfo& win_info) noexcept ;

        [[nodiscard]] static std::optional<int> HasPinfu(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasPureDoubleChis(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasTwicePureDoubleChis(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasSevenPairs(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllPons(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasPureStraight(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasMixedTripleChis(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasTriplePons(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasOutsideHand(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
        [[nodiscard]] static std::optional<int> HasTerminalsInAllSets(
                const WinningInfo& win_info,
                const std::vector<TileTypeCount>& closed_sets,
                const std::vector<TileTypeCount>& opened_sets,
                const std::vector<TileTypeCount>& heads) noexcept ;
    };
} // namespace mj

#endif //MAHJONG_YAKU_EVALUATOR_H
