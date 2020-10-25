#ifndef MAHJONG_GRPC_SERVER_AGENT_EXAMPLE_H
#define MAHJONG_GRPC_SERVER_AGENT_EXAMPLE_H

#include "mj.grpc.pb.h"

#include "grpc_server_agent.h"

namespace mj
{
    class GrpcServerAgentExampleImpl final : public mjproto::Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
    };


    class GrpcServerAgentExample final : public GrpcServerAgent
    {
    public:
        GrpcServerAgentExample();
        ~GrpcServerAgentExample() final = default;
        void RunServer(const std::string &socket_address) final ;
    private:
        std::unique_ptr<grpc::Service> agent_impl_;
    };

}  // namespace mj

#endif //MAHJONG_GRPC_SERVER_AGENT_EXAMPLE_H
