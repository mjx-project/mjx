#ifndef MAHJONG_AGENT_INTERFACE_H
#define MAHJONG_AGENT_INTERFACE_H

#include <grpcpp/channel.h>
#include "action.h"
#include "observation.h"
#include "mj.grpc.pb.h"
#include "agent.h"

namespace mj
{
    class AgentInterface
    {
    public:
        AgentInterface() = default;
        virtual ~AgentInterface() = default;
        [[nodiscard]] virtual mjproto::Action TakeAction(Observation &&observation) const = 0;
        [[nodiscard]] virtual PlayerId player_id() const = 0;
    };

    class AgentInterfaceGrpc final: public AgentInterface
    {
    public:
        AgentInterfaceGrpc() = default;  // will make invalid object
        explicit AgentInterfaceGrpc(const std::shared_ptr<grpc::Channel>& channel);
        ~AgentInterfaceGrpc() final = default;
        [[nodiscard]] mjproto::Action TakeAction(Observation &&observation) const final ;
        [[nodiscard]] PlayerId player_id() const final;
    private:
        std::unique_ptr<mjproto::Agent::Stub> stub_;
    };

    class AgentInterfaceLocal final: public AgentInterface
    {
    public:
        AgentInterfaceLocal() = default;  // will make invalid object
        explicit AgentInterfaceLocal(std::unique_ptr<Agent> agent);
        ~AgentInterfaceLocal() final = default;
        [[nodiscard]] mjproto::Action TakeAction(Observation &&observation) const final;
        [[nodiscard]] PlayerId player_id() const final;
    private:
        std::unique_ptr<Agent> agent_;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_INTERFACE_H
