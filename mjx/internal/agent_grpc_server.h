#ifndef MAHJONG_AGENT_GRPC_SERVER_H
#define MAHJONG_AGENT_GRPC_SERVER_H

#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/observation.h"
#include "mjx/internal/strategy_rule_based.h"

namespace mjx::internal {
class AgentGrpcServer {
 public:
  static void RunServer(std::unique_ptr<Strategy> strategy,
                        const std::string& socket_address);
};

class AgentGrpcServerImpl final : public mjxproto::Agent::Service {
 public:
  explicit AgentGrpcServerImpl(std::unique_ptr<Strategy> strategy);
  ~AgentGrpcServerImpl() final = default;
  grpc::Status TakeAction(grpc::ServerContext* context,
                          const mjxproto::Observation* request,
                          mjxproto::Action* reply) final;

 private:
  // Agent logic
  std::unique_ptr<Strategy> strategy_;
};
}  // namespace mjx::internal

#endif  // MAHJONG_AGENT_GRPC_SERVER_H
