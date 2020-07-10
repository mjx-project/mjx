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
        YakuEvaluator() = delete;
        [[nodiscard]] static WinningScore Eval(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static bool Has(const WinningInfo& win_info) noexcept ;    // 上がりの形になっていれば良い.
        [[nodiscard]] static bool CanWin(const WinningInfo& win_info) noexcept ;    // 役がないとダメ.

    private:
        [[nodiscard]] static const WinningHandCache& win_cache();

        static void JudgeYakuman(
                const WinningInfo& win_info,
                WinningScore& score) noexcept ;

        static void JudgeSimpleYaku(
                const WinningInfo& win_info,
                WinningScore& score) noexcept ;

        static int TotalFan(const std::map<Yaku,int>& yaku) noexcept ;
        [[nodiscard]] static std::tuple<std::map<Yaku,int>,std::vector<TileTypeCount>,std::vector<TileTypeCount>>
        MaximizeTotalFan(const WinningInfo& win_info) noexcept ;

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
        [[nodiscard]] static std::optional<int> HasRiichi(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasAfterKan(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasRobbingKan(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasBottomOfTheSea(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasBottomOfTheRiver(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasAllSimples(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasWhiteDragon(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasGreenDragon(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasRedDragon(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasSeatWindEast(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasSeatWindSouth(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasSeatWindWest(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasSeatWindNorth(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasPrevalentWindEast(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasPrevalentWindSouth(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasPrevalentWindWest(const WinningInfo& win_info) noexcept ;
        [[nodiscard]] static std::optional<int> HasPrevalentWindNorth(const WinningInfo& win_info) noexcept ;
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
