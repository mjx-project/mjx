#ifndef MJX_REPO_AGENT_LOCAL_H
#define MJX_REPO_AGENT_LOCAL_H

#include "mjx/internal/action.h"
#include "mjx/internal/agent.h"
#include "mjx/internal/observation.h"
#include "mjx/internal/strategy.h"

namespace mjx::internal {
class AgentLocal final : public Agent {
 public:
  AgentLocal() = default;
  AgentLocal(PlayerId player_id, std::shared_ptr<Strategy> strategy);
  ~AgentLocal() final = default;
  [[nodiscard]] mjxproto::Action TakeAction(
      Observation &&observation) const final;

 private:
  // Agent logic
  std::shared_ptr<Strategy> strategy_;
};
}  // namespace mjx::internal

#endif  // MJX_REPO_AGENT_LOCAL_H
