#include "mjx/agent.h"

#include "mjx/internal/observation.h"
#include "mjx/internal/strategy_rule_based.h"

namespace mjx {

Agent::Agent(const std::string& strategy) : strategy(strategy) {}

Action Agent::TakeAction(const Observation& obs) {
  if (strategy == "rule_based") {
    return Agent::TakeActionRuleBased(obs);
  }
}

Action Agent::TakeActionRuleBased(const Observation& obs) {
  return Action(internal::StrategyRuleBased().TakeAction(
      internal::Observation(obs.proto())));
}

}  // namespace mjx
