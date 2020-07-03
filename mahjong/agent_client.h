#ifndef MAHJONG_AGENT_CLIENT_H
#define MAHJONG_AGENT_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <mahjong.grpc.pb.h>
#include "action.h"
#include "observation.h"

namespace mj
{
    class AgentClient
    {
    public:
        AgentClient() = default;
        explicit AgentClient(std::shared_ptr<grpc::Channel> channel);
        virtual ~AgentClient() = default;
        [[nodiscard]] virtual Action TakeAction(std::unique_ptr<Observation> observation) const;
    private:
        std::unique_ptr<mjproto::Agent::Stub> stub_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_CLIENT_H
