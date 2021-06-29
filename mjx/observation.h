#ifndef MAHJONG_OBSERVATION_H
#define MAHJONG_OBSERVATION_H

#include <array>
#include <utility>

#include "action.h"
#include "internal/hand.h"
#include "internal/mjx.pb.h"

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

  [[nodiscard]] std::vector<float> ToFeature(std::string version) const;

 private:
  // TODO: remove friends and use proto()
  friend class State;
  mjxproto::Observation proto_ = mjxproto::Observation{};

  [[nodiscard]] std::vector<float> small_v0() const;
};
}  // namespace mjx::internal

#endif  // MAHJONG_OBSERVATION_H
