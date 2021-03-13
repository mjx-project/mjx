#ifndef MJX_REPO_AGENT_LOCAL_H
#define MJX_REPO_AGENT_LOCAL_H

#include "action.h"
#include "agent.h"
#include "observation.h"
#include "strategy.h"

namespace mjx {
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
}  // namespace mjx

#endif  // MJX_REPO_AGENT_LOCAL_H
