#include "agent_local.h"

namespace mjx {
AgentLocal::AgentLocal(PlayerId player_id, std::unique_ptr<Strategy> strategy)
    : Agent(std::move(player_id)), strategy_(std::move(strategy)) {}

mjxproto::Action AgentLocal::TakeAction(Observation &&observation) const {
  return strategy_->TakeAction(std::forward<Observation>(observation));
}
}  // namespace mjx
