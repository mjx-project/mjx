#ifndef MAHJONG_GRPC_CLIENT_AGENT_H
#define MAHJONG_GRPC_CLIENT_AGENT_H

#include <grpcpp/grpcpp.h>
#include "mj.grpc.pb.h"
#include "agent.h"
#include "action.h"
#include "observation.h"

namespace mj
{
    class GrpcClientAgent final: public Agent
    {
    public:
        GrpcClientAgent() = default;
        GrpcClientAgent(PlayerId player_id, const std::shared_ptr<grpc::Channel>& channel);
        ~GrpcClientAgent() final = default;
        [[nodiscard]] Action TakeAction(Observation &&observation) const final ;
    private:
        std::unique_ptr<mjproto::Agent::Stub> stub_;
    };
}

#endif //MAHJONG_GRPC_CLIENT_AGENT_H
