#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include "mjx/action.h"
#include "mjx/observation.h"

namespace mjx {
class Agent {
 public:
  virtual ~Agent() = default;
  [[nodiscard]] virtual Action Act(
      const Observation& observation) const noexcept = 0;
};

class RandomAgent final : public Agent {
 public:
  [[nodiscard]] virtual Action Act(
      const Observation& observation) const noexcept;
};

class GrpcAgent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] virtual Action Act(
      const Observation& observation) const noexcept;

 private:
  std::shared_ptr<mjxproto::Agent::Stub> stub_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
