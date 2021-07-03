#ifndef MAHJONG_AGENT_GRPC_CLIENT_H
#define MAHJONG_AGENT_GRPC_CLIENT_H

#include <grpcpp/grpcpp.h>

#include "mjx/internal/action.h"
#include "mjx/internal/agent.h"
#include "mjx/internal/mjx.grpc.pb.h"
#include "mjx/internal/observation.h"

namespace mjx::internal {
class AgentGrpcClient final : public Agent {
 public:
  AgentGrpcClient() = default;
  AgentGrpcClient(PlayerId player_id,
                  const std::shared_ptr<grpc::Channel>& channel);
  ~AgentGrpcClient() final = default;
  [[nodiscard]] mjxproto::Action TakeAction(
      Observation&& observation) const final;

 private:
  std::unique_ptr<mjxproto::Agent::Stub> stub_;
};
}  // namespace mjx::internal

#endif  // MAHJONG_AGENT_GRPC_CLIENT_H
