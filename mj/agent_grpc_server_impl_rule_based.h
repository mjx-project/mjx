#ifndef MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H
#define MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H

#include "mj.grpc.pb.h"
#include "agent_grpc_server.h"

namespace mj
{
    class AgentGrpcServerImplRuleBased final : public mjproto::Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_GRPC_SERVER_IMPL_RULE_BASED_H
