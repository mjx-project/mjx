#ifndef MAHJONG_STRATEGY_H
#define MAHJONG_STRATEGY_H

#include "mjx/internal/observation.h"

namespace mjx::internal {
class Strategy {
 public:
  virtual ~Strategy() = default;
  [[nodiscard]] virtual std::vector<mjxproto::Action> TakeActions(
      std::vector<Observation> &&observations) const = 0;
  [[nodiscard]] virtual mjxproto::Action TakeAction(
      Observation &&observation) const = 0;
};
}  // namespace mjx::internal

#endif  // MAHJONG_STRATEGY_H
