#ifndef MAHJONG_AGENT_CLIENT_H
#define MAHJONG_AGENT_CLIENT_H

#include <grpcpp/grpcpp.h>
#include "mj.grpc.pb.h"
#include "action.h"
#include "observation.h"

namespace mj
{
    class Agent
    {
    public:
        Agent() = default;  // generate invalid object
        explicit Agent(PlayerId player_id);
        explicit Agent(PlayerId player_id, const std::shared_ptr<grpc::Channel>& channel);
        virtual ~Agent() = default;
        [[nodiscard]] virtual Action TakeAction(Observation &&observation) const;
        [[nodiscard]] PlayerId player_id() const;
    private:
        PlayerId player_id_;
        std::unique_ptr<mjproto::Agent::Stub> stub_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_CLIENT_H
