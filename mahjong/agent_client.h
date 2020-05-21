#ifndef MAHJONG_AGENT_CLIENT_H
#define MAHJONG_AGENT_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <mahjong.grpc.pb.h>
#include "action.h"

namespace mj
{
    class AgentClient
    {
    public:
        AgentClient(std::shared_ptr<grpc::Channel> channel);
        std::unique_ptr<Action> TakeAction();
    private:
        std::unique_ptr<Agent::Stub> stub_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_CLIENT_H
