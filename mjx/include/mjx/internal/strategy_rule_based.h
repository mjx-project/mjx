#ifndef MAHJONG_STRATEGY_RULE_BASED_H
#define MAHJONG_STRATEGY_RULE_BASED_H

#include "mjx/internal/observation.h"
#include "mjx/internal/strategy.h"

namespace mjx::internal {
class StrategyRuleBased final : public Strategy {
 public:
  ~StrategyRuleBased() final = default;
  [[nodiscard]] std::vector<mjxproto::Action> TakeActions(
      std::vector<Observation> &&observations) const final;
  [[nodiscard]] mjxproto::Action TakeAction(
      Observation &&observation) const final;

 private:
  template <typename RandomGenerator>
  static Tile SelectDiscard(std::vector<Tile> &discard_candidates,
                            const Hand &curr_hand, RandomGenerator &g);
};
}  // namespace mjx::internal

#endif  // MAHJONG_STRATEGY_RULE_BASED_H
