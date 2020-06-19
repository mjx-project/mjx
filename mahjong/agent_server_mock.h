#ifndef MAHJONG_AGENT_SERVER_MOCK_H
#define MAHJONG_AGENT_SERVER_MOCK_H

#include "mahjong.grpc.pb.h"

#include "agent_server.h"

namespace mj
{
    class MockAgentServiceImpl final : public Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const ActionRequest* request, ActionResponse* reply) final ;
    };


    class MockAgentServer final : public AgentServer
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
