#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/observation.h"
#include "mjx/action.h"

namespace mjx {
class Agent {
 public:
  Agent() = default;
  explicit Agent(const std::string& strategy);
  Action TakeAction(const Observation& obs);
 private:
  Action TakeActionRuleBased(const Observation& obs);
  const std::string strategy;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
