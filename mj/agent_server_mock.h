#ifndef MAHJONG_AGENT_SERVER_MOCK_H
#define MAHJONG_AGENT_SERVER_MOCK_H

#include "mj.grpc.pb.h"

#include "grpc_server_agent.h"

namespace mj
{
    class MockAgentServiceImpl final : public mjproto::Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
    };


    class MockAgentServer final : public GrpcServerAgent
    {
    public:
        MockAgentServer();
        ~MockAgentServer() final = default;
        void RunServer(const std::string &socket_address) final ;
    private:
        std::unique_ptr<grpc::Service> agent_impl_;
    };

}  // namespace mj

#endif //MAHJONG_AGENT_SERVER_MOCK_H
