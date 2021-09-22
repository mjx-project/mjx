#include "mjx/internal/agent_local.h"

#include <utility>

namespace mjx::internal {
AgentLocal::AgentLocal(PlayerId player_id, std::shared_ptr<Strategy> strategy)
    : Agent(std::move(player_id)), strategy_(std::move(strategy)) {}

mjxproto::Action AgentLocal::TakeAction(Observation &&observation) const {
  return strategy_->TakeAction(std::forward<Observation>(observation));
}
}  // namespace mjx::internal
