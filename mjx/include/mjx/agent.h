#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include "mjx/action.h"
#include "mjx/internal/utils.h"
#include "mjx/observation.h"

namespace mjx {

class Agent {
 public:
  virtual ~Agent() {}
  [[nodiscard]] virtual mjx::Action Act(const Observation& observation) const noexcept = 0;
  void Serve(const std::string& socket_address) const noexcept;
};

class RandomAgent : public Agent {
 public:
  [[nodiscard]] mjx::Action Act(const Observation& observation) const noexcept override;
};

class GrpcAgent : public Agent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] mjx::Action Act(const Observation& observation) const noexcept override;

 private:
  std::shared_ptr<mjxproto::Agent::Stub> stub_;
};

class AgentGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentGrpcServerImpl(const Agent* agent);
  ~AgentGrpcServerImpl() final = default;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

 private:
  const Agent* agent_;
};
}  // namespace mjx

#endif  // MJX_PROJECT_AGENT_H
