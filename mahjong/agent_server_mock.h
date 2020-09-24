#ifndef MAHJONG_AGENT_SERVER_MOCK_H
#define MAHJONG_AGENT_SERVER_MOCK_H

#include "mahjong.grpc.pb.h"

#include "agent_server.h"

namespace mj
{
    class MockAgentServiceImpl final : public mjproto::Agent::Service
    {
    public:
        grpc::Status TakeAction(grpc::ServerContext* context, const mjproto::Observation* request, mjproto::Action* reply) final ;
    };


    class MockAgentServer final : public AgentServer
    {
    public:
        MockAgentServer();
        ~MockAgentServer() final = default;
    };

}  // namespace mj

#endif //MAHJONG_AGENT_SERVER_MOCK_H
