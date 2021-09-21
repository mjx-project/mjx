#ifndef MJX_PROJECT_AGENT_H
#define MJX_PROJECT_AGENT_H

#include <grpcpp/grpcpp.h>

#include "mjx/action.h"
#include "mjx/internal/utils.h"
#include "mjx/observation.h"

namespace mjx {

class Agent {
 public:
  virtual ~Agent() {}
  [[nodiscard]] virtual mjx::Action Act(
      const Observation& observation) const noexcept = 0;
  void Serve(const std::string& socket_address) noexcept;
  void Wait() const noexcept;
  void Shutdown() const noexcept;

 private:
  std::unique_ptr<grpc::Server> server_;
};

// Agent that acts randomly but in the reproducible way.
// The same observation should return the same action.
// Only for debugging purpose.
class RandomDebugAgent : public Agent {
 public:
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;
};

class GrpcAgent : public Agent {
 public:
  explicit GrpcAgent(const std::string& socket_address);
  [[nodiscard]] mjx::Action Act(
      const Observation& observation) const noexcept override;

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
