#ifndef MAHJONG_AGENT_SERVER_H
#define MAHJONG_AGENT_SERVER_H

#include "mj.grpc.pb.h"

namespace mj
{
    class AgentServer
    {
    public:
        virtual ~AgentServer() = default;
        virtual void RunServer(const std::string &socket_address) = 0;
    };
}  // namespace mj

#endif //MAHJONG_AGENT_SERVER_H
