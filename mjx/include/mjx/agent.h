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
  void Serve(const std::string& socket_address) noexcept;
};

class RandomAgent final : public Agent {
 public:
  [[nodiscard]] virtual Action Act(
      const Observation& observation) const noexcept;
};

class GrpcAgent final : public Agent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] virtual Action Act(
      const Observation& observation) const noexcept;

 private:
  std::shared_ptr<mjxproto::Agent::Stub> stub_;
};

class AgentGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentGrpcServerImpl(Agent* agent);
  ~AgentGrpcServerImpl() final = default;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

 private:
  Agent* agent_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
