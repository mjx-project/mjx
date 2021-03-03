#ifndef MAHJONG_AGENT_NONBATCH_GRPC_SERVER_H
#define MAHJONG_AGENT_NONBATCH_GRPC_SERVER_H

#include "mjx.grpc.pb.h"
#include "strategy_rule_based.h"
#include "observation.h"

namespace mjx
{
    class AgentGrpcServer
    {
    public:
        static void RunServer(std::unique_ptr<Strategy> strategy, const std::string &socket_address);
    };

    class AgentGrpcServerImpl final : public mjxproto::Agent::Service
    {
    public:
        explicit AgentGrpcServerImpl(std::unique_ptr<Strategy> strategy);
        ~AgentGrpcServerImpl() final = default;
        grpc::Status TakeAction(grpc::ServerContext* context, const mjxproto::Observation* request, mjxproto::Action* reply) final ;
    private:
        // Agent logic
        std::unique_ptr<Strategy> strategy_;
    };
}  // namespace mjx

#endif //MAHJONG_AGENT_NONBATCH_GRPC_SERVER_H
