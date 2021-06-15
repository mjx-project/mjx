#ifndef MAHJONG_STRATEGY_H
#define MAHJONG_STRATEGY_H

#include "observation.h"
#include <optional>

namespace mjx::internal {
class Strategy {
 public:
  virtual ~Strategy() = default;
  [[nodiscard]] virtual std::vector<std::optional<mjxproto::Action>>
  TakeActions(
      std::vector<Observation> &&observations) const = 0;
  [[nodiscard]] virtual std::optional<mjxproto::Action> TakeAction(
      Observation &&observation) const = 0;
};
}  // namespace mjx::internal

#endif  // MAHJONG_STRATEGY_H
