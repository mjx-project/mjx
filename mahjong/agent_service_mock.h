#ifndef MAHJONG_AGENT_SERVICE_MOCK_H
#define MAHJONG_AGENT_SERVICE_MOCK_H

#include "mahjong.grpc.pb.h"

namespace mj
{
    class MockAgentService final : public Agent::Service
    {
        grpc::Status TakeAction(grpc::ServerContext* context, const ActionRequest* request, ActionResponse* reply) final ;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_SERVICE_MOCK_H
