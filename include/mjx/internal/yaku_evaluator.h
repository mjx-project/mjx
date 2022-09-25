#ifndef MAHJONG_YAKU_EVALUATOR_H
#define MAHJONG_YAKU_EVALUATOR_H

#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "mjx/internal/types.h"
#include "mjx/internal/win_cache.h"
#include "mjx/internal/win_info.h"
#include "mjx/internal/win_score.h"

namespace mjx::internal {
class YakuEvaluator {
 public:
  YakuEvaluator() = delete;
  [[nodiscard]] static WinScore Eval(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool CanWin(
      const WinInfo& win_info) noexcept;  // 役がないとダメ.

 private:
  [[nodiscard]] static const WinHandCache& win_cache();

  static void JudgeYakuman(const WinInfo& win_info, WinScore& score) noexcept;

  static void JudgeSimpleYaku(const WinInfo& win_info,
                              WinScore& score) noexcept;

  static void JudgeDora(const WinInfo& win_info, WinScore& score) noexcept;

  static int TotalFan(const std::map<Yaku, int>& yaku) noexcept;
  [[nodiscard]] static std::tuple<std::map<Yaku, int>, int,
                                  std::vector<TileTypeCount>,
                                  std::vector<TileTypeCount>>
  MaximizeTotalFan(const WinInfo& win_info) noexcept;

  [[nodiscard]] static int CalculateFu(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& heads,
      const std::map<Yaku, int>& yakus) noexcept;

  [[nodiscard]] static bool HasBlessingOfHeaven(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasBlessingOfEarth(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasBigThreeDragons(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasAllHonours(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasAllGreen(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasAllTerminals(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasBigFourWinds(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasLittleFourWinds(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasThirteenOrphans(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasCompletedThirteenOrphans(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasNineGates(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasPureNineGates(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasFourKans(const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasFourConcealedPons(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static bool HasCompletedFourConcealedPons(
      const WinInfo& win_info) noexcept;

  [[nodiscard]] static std::optional<int> HasRedDora(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasDora(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasReversedDora(
      const WinInfo& win_info) noexcept;

  [[nodiscard]] static std::optional<int> HasFullyConcealdHand(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasRiichi(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasDoubleRiichi(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasAfterKan(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasRobbingKan(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasBottomOfTheSea(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasBottomOfTheRiver(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasIppatsu(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasAllSimples(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasWhiteDragon(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasGreenDragon(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasRedDragon(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasSeatWindEast(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasSeatWindSouth(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasSeatWindWest(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasSeatWindNorth(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasPrevalentWindEast(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasPrevalentWindSouth(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasPrevalentWindWest(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasPrevalentWindNorth(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasAllTermsAndHonours(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasHalfFlush(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasFullFlush(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasThreeKans(
      const WinInfo& win_info) noexcept;
  [[nodiscard]] static std::optional<int> HasLittleThreeDragons(
      const WinInfo& win_info) noexcept;

  [[nodiscard]] static std::optional<int> HasPinfu(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasPureDoubleChis(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasTwicePureDoubleChis(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasSevenPairs(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasAllPons(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasPureStraight(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasMixedTripleChis(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasTriplePons(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasOutsideHand(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasTerminalsInAllSets(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
  [[nodiscard]] static std::optional<int> HasThreeConcealdPons(
      const WinInfo& win_info, const std::vector<TileTypeCount>& closed_sets,
      const std::vector<TileTypeCount>& opened_sets,
      const std::vector<TileTypeCount>& heads) noexcept;
};
}  // namespace mjx::internal

#endif  // MAHJONG_YAKU_EVALUATOR_H
