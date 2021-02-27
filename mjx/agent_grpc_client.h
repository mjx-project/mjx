#ifndef MAHJONG_AGENT_GRPC_CLIENT_H
#define MAHJONG_AGENT_GRPC_CLIENT_H

#include <grpcpp/grpcpp.h>
#include "mjx.grpc.pb.h"
#include "agent.h"
#include "action.h"
#include "observation.h"

namespace mjx
{
    class AgentGrpcClient final: public Agent
    {
    public:
        AgentGrpcClient() = default;
        AgentGrpcClient(PlayerId player_id, const std::shared_ptr<grpc::Channel>& channel);
        ~AgentGrpcClient() final = default;
        [[nodiscard]] mjproto::Action TakeAction(Observation &&observation) const final ;
    private:
        std::unique_ptr<mjproto::Agent::Stub> stub_;
    };
}

#endif //MAHJONG_AGENT_GRPC_CLIENT_H
