#include "mjx/internal/agent.h"

#include <utility>

#include "mjx/internal/utils.h"

namespace mjx::internal {
Agent::Agent(PlayerId player_id) : player_id_(std::move(player_id)) {}

PlayerId Agent::player_id() const {
  Assert(!player_id_.empty());
  return player_id_;
}
}  // namespace mjx::internal
