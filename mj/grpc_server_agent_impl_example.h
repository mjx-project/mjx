#ifndef MAHJONG_GRPC_SERVER_AGENT_IMPL_EXAMPLE_H
#define MAHJONG_GRPC_SERVER_AGENT_IMPL_EXAMPLE_H

#include "mj.grpc.pb.h"
#include "grpc_server_agent.h"

namespace mj
{
    class GrpcServerAgentImplExample final : public mjproto::Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
    };
}  // namespace mj

#endif //MAHJONG_GRPC_SERVER_AGENT_IMPL_EXAMPLE_H
