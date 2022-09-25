#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <array>
#include <optional>
#include <utility>

#include "mjx/internal/action.h"
#include "mjx/internal/hand.h"
#include "mjx/internal/mjx.pb.h"

namespace mjx::internal {
class Observation {
 public:
  Observation() = default;
  Observation(const mjxproto::Observation& proto);
  Observation(AbsolutePos who, const mjxproto::State& state);

  AbsolutePos who() const;
  [[nodiscard]] bool has_legal_action() const;
  [[nodiscard]] std::vector<mjxproto::Action> legal_actions() const;
  [[nodiscard]] std::vector<std::pair<Tile, bool>> possible_discards() const;
  Hand initial_hand() const;
  Hand current_hand() const;
  std::string ToJson() const;
  const mjxproto::Observation& proto() const;
  [[nodiscard]] std::vector<mjxproto::Event> EventHistory() const;

  void add_legal_action(mjxproto::Action&& legal_action);
  void add_legal_actions(const std::vector<mjxproto::Action>& legal_actions);

  [[nodiscard]] static std::vector<mjxproto::Action> GenerateLegalActions(
      const mjxproto::Observation& observation);

  [[nodiscard]] std::vector<std::vector<int>> ToFeaturesSmallV0() const;
  [[nodiscard]] std::vector<std::vector<int>> ToFeaturesHan22V0() const;

 private:
  // TODO: remove friends and use proto()
  friend class State;
  mjxproto::Observation proto_ = mjxproto::Observation{};

  // 次のブロックは主にObservationだけからでもlegal
  // actionを生成できるようにするための機能
  // internal::Stateの内部状態を使わずに同じ計算をしている想定
  // ただし、eventsをなめるので遅かったりする
  [[nodiscard]] static bool HasDrawLeft(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool HasNextDrawLeft(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool RequireKanDraw(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool CanRon(AbsolutePos who,
                                   const mjxproto::Observation& observation);
  [[nodiscard]] static bool CanTsumo(AbsolutePos who,
                                     const mjxproto::Observation& observation);
  [[nodiscard]] static std::optional<Tile> TargetTile(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static AbsolutePos dealer(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static Wind prevalent_wind(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool IsIppatsu(
      AbsolutePos who, const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool IsRobbingKan(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool IsFirstTurnWithoutOpen(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool IsFourKanNoWinner(
      const mjxproto::PublicObservation& public_observation);
  [[nodiscard]] static bool CanRiichi(AbsolutePos who,
                                      const mjxproto::Observation& observation);
  [[nodiscard]] static bool IsRoundOver(
      const mjxproto::PublicObservation& public_observation);
};
}  // namespace mjx::internal

#endif  // MAHJONG_OBSERVATION_H
